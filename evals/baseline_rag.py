"""A stateless retrieval-augmented chatbot baseline with no tools or memory."""

from pydantic import BaseModel

from kb.search import search_kb
from llm.client import call_llm_json

BASELINE_PROMPT = """Answer the customer's question using only the retrieved support
articles. This baseline has no access to customer tools, conversation memory, or
action tools. If the articles do not answer the question, say what information is
missing instead of inventing a policy or claiming an action was taken."""


class BaselineAnswer(BaseModel):
	"""Validated answer from the retrieval-only baseline."""

	answer: str


def answer_query(query: str, k: int = 4) -> str:
	"""Search the KB and generate one answer without tools, memory, or actions."""
	articles = search_kb(query, k=k)
	context = "\n\n".join(
		f"Article: {article.get('title', '')}\n"
		f"Workflow: {article.get('workflow_id', '')}\n"
		f"Content: {article.get('content', article.get('text', ''))}"
		for article in articles
	)
	user_prompt = f"Customer question:\n{query}\n\nRetrieved articles:\n{context or 'No articles found.'}"
	result = call_llm_json(BASELINE_PROMPT, user_prompt, BaselineAnswer, "baseline_rag")
	return result.answer