"""Retrieve relevant articles and select one supported workflow."""

import json
import logging
from typing import Any

from agents.schemas import (
	CustomerContext,
	IntentResult,
	KBResult,
	KnowledgeArticle,
	WorkflowSelection,
)
from kb.search import search_kb
from llm.client import call_llm_json

WORKFLOW_SELECTION_PROMPT = """Choose the single best workflow_id from the retrieved
knowledge candidates for the customer's issue. Select only a workflow shown in the
candidates, or 'none' when none applies. Base confidence on relevance and article
reliability; use a value below 0.5 when evidence is weak. Do not invent workflows."""
WORKFLOW_BY_INTENT = {
	"wrong_item": "wrong_item",
	"damaged_item": "damaged_item",
	"missing_item": "missing_item",
	"order_status": "order_status",
	"payment_failure": "payment_deducted",
	"refund_request": "refund_request",
	"cancel_order": "cancel_order",
	"login_issue": "account_locked",
	"subscription_issue": "subscription_sync",
	"fraud_complaint": "fraud_escalate",
	"legal_request": "legal_escalate",
	"data_breach": "legal_escalate",
	"general_query": "general_info",
}
logger = logging.getLogger(__name__)


def _article_from_result(result: dict[str, Any], index: int) -> KnowledgeArticle:
	"""Convert one search hit into its validated agent schema."""
	return KnowledgeArticle(
		article_id=str(result.get("article_id", result.get("id", f"article-{index}"))),
		title=str(result.get("title", "Untitled article")),
		category=str(result.get("category", "general")),
		reliability_score=float(result.get("reliability_score", 0.0)),
		workflow_id=str(result.get("workflow_id", "none")),
		content=str(result.get("content", result.get("text", ""))),
		similarity=float(result.get("similarity", 0.0)),
		ranking_score=float(result.get("ranking_score", 0.0)),
	)


def select_workflow(
	intent: str | IntentResult,
	context: CustomerContext | None,
	query: str,
) -> KBResult:
	"""Search the KB and select its strongest applicable workflow with Gemini."""
	search_results = search_kb(query, k=4)
	articles = [_article_from_result(result, index) for index, result in enumerate(search_results)]
	if not articles:
		return KBResult(articles=[], selected_workflow="none", confidence=0.0)

	intent_name = intent.intent if isinstance(intent, IntentResult) else intent
	expected_workflow = WORKFLOW_BY_INTENT.get(intent_name)
	if expected_workflow is not None:
		articles = [article for article in articles if article.workflow_id == expected_workflow]
		if not articles:
			return KBResult(articles=[], selected_workflow=expected_workflow, confidence=1.0)
	candidate_workflows = {article.workflow_id for article in articles}
	context_summary = context.summary if context is not None else "No customer context provided."
	prompt_data = {
		"intent": intent_name,
		"customer_context": context_summary,
		"query": query,
		"candidates": [article.model_dump(mode="json") for article in articles],
	}
	try:
		selection = call_llm_json(
			WORKFLOW_SELECTION_PROMPT,
			json.dumps(prompt_data, ensure_ascii=True),
			WorkflowSelection,
			"knowledge",
		)
	except Exception as error:
		logger.warning("Workflow selection failed; using top search result: %s", error)
		best_article = max(articles, key=lambda article: article.ranking_score)
		confidence = min(1.0, max(0.0, best_article.ranking_score))
		return KBResult(
			articles=articles,
			selected_workflow=(best_article.workflow_id if confidence >= 0.5 else "none"),
			confidence=confidence,
		)

	if selection.workflow_id not in candidate_workflows:
		logger.warning("Gemini selected a workflow absent from the retrieved candidates.")
		best_article = max(articles, key=lambda article: article.ranking_score)
		workflow_id = best_article.workflow_id
		confidence = min(selection.confidence, best_article.ranking_score)
	else:
		workflow_id = selection.workflow_id
		confidence = selection.confidence
	if confidence < 0.5:
		workflow_id = "none"
	return KBResult(
		articles=articles,
		selected_workflow=workflow_id,
		confidence=confidence,
	)