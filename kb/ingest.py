"""Parse, chunk, embed, and persist markdown knowledge articles."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_DIRECTORY = PROJECT_ROOT / "data" / "kb"
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma"
JSON_INDEX_PATH = PROJECT_ROOT / "data" / "kb_vectors.json"
BACKEND_MARKER_PATH = PROJECT_ROOT / "data" / "kb_vector_backend.json"
COLLECTION_NAME = "resolveai_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHUNK_CHARACTERS = 1200
CHUNK_OVERLAP_CHARACTERS = 160

logger = logging.getLogger(__name__)
_SENTENCE_EMBEDDER: Any | None = None


@dataclass(frozen=True)
class KBArticle:
	"""Validated article fields parsed from Markdown front matter."""

	article_id: str
	title: str
	category: str
	reliability_score: float
	workflow_id: str
	content: str


def load_articles(articles_directory: Path | None = None) -> list[KBArticle]:
	"""Load and validate every Markdown article in the knowledge directory."""
	directory = articles_directory or ARTICLES_DIRECTORY
	articles: list[KBArticle] = []
	for article_path in sorted(directory.rglob("*.md")):
		markdown = article_path.read_text(encoding="utf-8")
		parts = markdown.split("---", 2)
		if len(parts) != 3 or parts[0].strip():
			raise ValueError(f"{article_path.name} must start with YAML front matter.")
		metadata = yaml.safe_load(parts[1])
		if not isinstance(metadata, dict):
			raise ValueError(f"{article_path.name} has invalid YAML front matter.")
		if metadata.get("status") == "rejected":
			continue
		required_fields = {"title", "category", "reliability_score", "workflow_id"}
		missing_fields = required_fields - metadata.keys()
		if missing_fields:
			raise ValueError(
				f"{article_path.name} is missing front matter: "
				f"{', '.join(sorted(missing_fields))}."
			)
		reliability_score = float(metadata["reliability_score"])
		if not 0.0 <= reliability_score <= 1.0:
			raise ValueError(f"{article_path.name} reliability_score must be between 0 and 1.")
		content = parts[2].strip()
		if not content:
			raise ValueError(f"{article_path.name} has no article content.")
		articles.append(
			KBArticle(
				article_id=article_path.stem,
				title=str(metadata["title"]),
				category=str(metadata["category"]),
				reliability_score=reliability_score,
				workflow_id=str(metadata["workflow_id"]),
				content=content,
			)
		)
	if not articles:
		raise ValueError(f"No Markdown knowledge articles found in {directory}.")
	return articles


def chunk_text(
	text: str,
	max_characters: int = MAX_CHUNK_CHARACTERS,
	overlap_characters: int = CHUNK_OVERLAP_CHARACTERS,
) -> list[str]:
	"""Split text into overlapping chunks while preferring paragraph boundaries."""
	if max_characters <= 0 or overlap_characters < 0:
		raise ValueError("Chunk size must be positive and overlap cannot be negative.")
	if overlap_characters >= max_characters:
		raise ValueError("Chunk overlap must be smaller than the chunk size.")
	remaining = text.strip()
	chunks: list[str] = []
	while remaining:
		if len(remaining) <= max_characters:
			chunks.append(remaining)
			break
		boundary = remaining.rfind("\n", 0, max_characters + 1)
		if boundary < max_characters // 2:
			boundary = max_characters
		chunk = remaining[:boundary].strip()
		if chunk:
			chunks.append(chunk)
		remaining = remaining[max(1, boundary - overlap_characters) :].strip()
	return chunks


def _document_chunks(articles: list[KBArticle]) -> list[dict[str, Any]]:
	"""Create text chunks with the article metadata needed for ranking."""
	chunks: list[dict[str, Any]] = []
	for article in articles:
		document = (
			f"{article.title}\nCategory: {article.category}\n"
			f"Workflow: {article.workflow_id}\n\n{article.content}"
		)
		for chunk_index, text in enumerate(chunk_text(document)):
			chunks.append(
				{
					"id": f"{article.article_id}-{chunk_index:03d}",
					"text": text,
					"metadata": {
						"article_id": article.article_id,
						"title": article.title,
						"category": article.category,
						"reliability_score": article.reliability_score,
						"workflow_id": article.workflow_id,
						"chunk_index": chunk_index,
					},
				}
			)
	return chunks


def _load_sentence_embedder() -> Any:
	"""Load and reuse the configured sentence-transformers model."""
	global _SENTENCE_EMBEDDER
	if _SENTENCE_EMBEDDER is None:
		from sentence_transformers import SentenceTransformer

		_SENTENCE_EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
	return _SENTENCE_EMBEDDER


def _encode_texts(
	texts: list[str], embedder: Any | None, task_type: str
) -> list[list[float]]:
	"""Embed text with Sentence Transformers or the Gemini fallback client."""
	if embedder is None:
		try:
			embedder = _load_sentence_embedder()
		except Exception:
			from llm.client import embed_texts

			return embed_texts(texts, task_type=task_type)
	embeddings = embedder.encode(
		texts,
		normalize_embeddings=True,
		convert_to_numpy=True,
	)
	return np.asarray(embeddings, dtype=float).tolist()


def _write_backend_marker(marker_path: Path, backend: str) -> None:
	"""Persist which vector backend contains the current index."""
	marker_path.parent.mkdir(parents=True, exist_ok=True)
	marker_path.write_text(json.dumps({"backend": backend}), encoding="utf-8")


def _ingest_chroma(
	chunks: list[dict[str, Any]],
	store_path: Path,
	marker_path: Path,
	embedder: Any | None,
) -> int:
	"""Replace the persistent Chroma collection with the current article chunks."""
	import chromadb

	store_path.mkdir(parents=True, exist_ok=True)
	client = chromadb.PersistentClient(path=str(store_path))
	collection = client.get_or_create_collection(
		name=COLLECTION_NAME,
		metadata={"hnsw:space": "cosine"},
	)
	existing_ids = collection.get(include=[])["ids"]
	if existing_ids:
		collection.delete(ids=existing_ids)
	embeddings = _encode_texts(
		[chunk["text"] for chunk in chunks], embedder, "RETRIEVAL_DOCUMENT"
	)
	collection.add(
		ids=[chunk["id"] for chunk in chunks],
		documents=[chunk["text"] for chunk in chunks],
		metadatas=[chunk["metadata"] for chunk in chunks],
		embeddings=embeddings,
	)
	_write_backend_marker(marker_path, "chroma")
	return len(chunks)


def _ingest_json(
	chunks: list[dict[str, Any]],
	index_path: Path,
	marker_path: Path,
	embedder: Any | None,
) -> int:
	"""Persist Gemini embeddings and metadata in a local JSON vector index."""
	index_path.parent.mkdir(parents=True, exist_ok=True)
	embeddings = _encode_texts(
		[chunk["text"] for chunk in chunks], embedder, "RETRIEVAL_DOCUMENT"
	)
	index = {
		"backend": "json",
		"records": [
			{
				"id": chunk["id"],
				"text": chunk["text"],
				"metadata": chunk["metadata"],
				"embedding": embedding,
			}
			for chunk, embedding in zip(chunks, embeddings, strict=True)
		],
	}
	index_path.write_text(json.dumps(index), encoding="utf-8")
	_write_backend_marker(marker_path, "json")
	return len(chunks)


def ingest_kb(
	articles_directory: Path | None = None,
	store_path: Path | None = None,
	index_path: Path | None = None,
	marker_path: Path | None = None,
	embedder: Any | None = None,
	backend: str = "auto",
) -> int:
	"""Ingest articles into Chroma, falling back to a persistent Gemini/JSON index."""
	if backend not in {"auto", "chroma", "json"}:
		raise ValueError("backend must be 'auto', 'chroma', or 'json'.")
	selected_store = store_path or CHROMA_DIRECTORY
	selected_index = index_path or JSON_INDEX_PATH
	selected_marker = marker_path or BACKEND_MARKER_PATH
	chunks = _document_chunks(load_articles(articles_directory))
	if backend != "json":
		try:
			return _ingest_chroma(chunks, selected_store, selected_marker, embedder)
		except Exception as error:
			if backend == "chroma":
				raise
			logger.warning(
				"Chroma/Sentence Transformers unavailable; using JSON fallback: %s", error
			)
	return _ingest_json(chunks, selected_index, selected_marker, embedder)


def main() -> None:
	"""Ingest the project knowledge articles into the persistent vector store."""
	logging.basicConfig(level=logging.INFO)
	chunk_count = ingest_kb()
	marker = json.loads(BACKEND_MARKER_PATH.read_text(encoding="utf-8"))
	print(f"Ingested {chunk_count} chunks using {marker['backend']} backend.")


if __name__ == "__main__":
	main()