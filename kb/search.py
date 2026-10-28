"""Search the persistent knowledge vector index with a lexical empty-index fallback."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from kb.ingest import (
	BACKEND_MARKER_PATH,
	CHROMA_DIRECTORY,
	COLLECTION_NAME,
	JSON_INDEX_PATH,
	_encode_texts,
	load_articles,
)

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
	"""Compute bounded cosine similarity for stored and query embeddings."""
	left_vector = np.asarray(left, dtype=float)
	right_vector = np.asarray(right, dtype=float)
	denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
	if denominator == 0.0:
		return 0.0
	return max(0.0, min(1.0, float(np.dot(left_vector, right_vector) / denominator)))


def _ranked_hit(
	metadata: dict[str, Any], text: str, similarity: float
) -> dict[str, Any]:
	"""Build the public article result and combined reliability ranking."""
	reliability = float(metadata.get("reliability_score", 0.0))
	return {
		"article_id": str(metadata.get("article_id", "")),
		"title": str(metadata.get("title", "")),
		"category": str(metadata.get("category", "general")),
		"content": text,
		"similarity": similarity,
		"reliability_score": reliability,
		"workflow_id": str(metadata.get("workflow_id", "none")),
		"ranking_score": similarity * reliability,
	}


def _lexical_search(query: str, category: str | None, k: int) -> list[dict[str, Any]]:
	"""Rank source articles lexically when no vector index has been ingested."""
	query_tokens = set(TOKEN_PATTERN.findall(query.lower()))
	if not query_tokens:
		return []
	hits: list[dict[str, Any]] = []
	for article in load_articles():
		if category and article.category.lower() != category.lower():
			continue
		content = f"{article.title}\n{article.category}\n{article.workflow_id}\n{article.content}"
		article_tokens = set(TOKEN_PATTERN.findall(content.lower()))
		intersection = len(query_tokens & article_tokens)
		union = len(query_tokens | article_tokens)
		similarity = intersection / union if union else 0.0
		if similarity > 0.0:
			hits.append(
				_ranked_hit(
					{
						"article_id": article.article_id,
						"title": article.title,
						"category": article.category,
						"reliability_score": article.reliability_score,
						"workflow_id": article.workflow_id,
					},
					article.content,
					similarity,
				)
			)
	return sorted(hits, key=lambda hit: hit["ranking_score"], reverse=True)[:k]


def _search_chroma(query: str, category: str | None, k: int) -> list[dict[str, Any]]:
	"""Query the persistent Chroma collection and convert distances to similarity."""
	import chromadb

	client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
	collection = client.get_collection(name=COLLECTION_NAME)
	query_embedding = _encode_texts([query], None, "RETRIEVAL_QUERY")[0]
	options: dict[str, Any] = {
		"query_embeddings": [query_embedding],
		"n_results": k,
		"include": ["documents", "metadatas", "distances"],
	}
	if category:
		options["where"] = {"category": category}
	result = collection.query(**options)
	documents = (result.get("documents") or [[]])[0]
	metadatas = (result.get("metadatas") or [[]])[0]
	distances = (result.get("distances") or [[]])[0]
	hits = [
		_ranked_hit(metadata, document, max(0.0, min(1.0, 1.0 - float(distance))))
		for document, metadata, distance in zip(documents, metadatas, distances)
	]
	return _best_chunk_per_article(hits, k)


def _search_json(query: str, category: str | None, k: int, index_path: Path) -> list[dict[str, Any]]:
	"""Search the persistent JSON embedding index using cosine similarity."""
	index = json.loads(index_path.read_text(encoding="utf-8"))
	records = index.get("records", [])
	if not records:
		return []
	query_embedding = _encode_texts([query], None, "RETRIEVAL_QUERY")[0]
	hits: list[dict[str, Any]] = []
	for record in records:
		metadata = record.get("metadata", {})
		if category and str(metadata.get("category", "")).lower() != category.lower():
			continue
		similarity = _cosine_similarity(query_embedding, record.get("embedding", []))
		hits.append(_ranked_hit(metadata, str(record.get("text", "")), similarity))
	return _best_chunk_per_article(hits, k)


def _best_chunk_per_article(hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
	"""Keep the best-matching chunk for each article and sort by combined score."""
	best_by_id: dict[str, dict[str, Any]] = {}
	for hit in hits:
		article_id = hit["article_id"]
		if article_id not in best_by_id or hit["ranking_score"] > best_by_id[article_id]["ranking_score"]:
			best_by_id[article_id] = hit
	return sorted(best_by_id.values(), key=lambda hit: hit["ranking_score"], reverse=True)[:k]


def search_kb(query: str, category: str | None = None, k: int = 4) -> list[dict[str, Any]]:
	"""Return KB articles ranked by similarity multiplied by reliability."""
	if k <= 0:
		return []
	if not query.strip():
		return []
	backend = None
	if BACKEND_MARKER_PATH.exists():
		try:
			backend = json.loads(BACKEND_MARKER_PATH.read_text(encoding="utf-8")).get("backend")
		except (OSError, json.JSONDecodeError) as error:
			logger.warning("Could not read KB backend marker: %s", error)
	if backend == "chroma" and CHROMA_DIRECTORY.exists():
		try:
			return _search_chroma(query, category, k)
		except Exception as error:
			logger.warning("Chroma search failed; trying available local index: %s", error)
	if JSON_INDEX_PATH.exists():
		try:
			return _search_json(query, category, k, JSON_INDEX_PATH)
		except Exception as error:
			logger.warning("JSON vector search failed; using lexical search: %s", error)
	return _lexical_search(query, category, k)