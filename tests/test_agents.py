"""Seeded, mocked tests for the implemented support agents."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agents import context_agent, intent_agent, knowledge_agent, troubleshooting_agent
from agents.schemas import (
	ContextSummary,
	ActionResult,
	ConversationState,
	CustomerContext,
	CSATResult,
	DiagnosisResponse,
	EscalationDecision,
	IntentResult,
	KBResult,
	Ticket,
)
from data.seed_data import seed_database
from db.models import Base


@pytest.fixture(scope="module")
def seeded_session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Session]:
	"""Create an isolated SQLite database containing C001 through C010."""
	database_path: Path = tmp_path_factory.mktemp("agent-data") / "support.db"
	database_url = f"sqlite:///{database_path.as_posix()}"
	seed_database(database_url)
	engine = create_engine(database_url, connect_args={"check_same_thread": False})
	with Session(engine) as session:
		yield session
	engine.dispose()


def _intent_result(
	intent: str = "order_status",
	priority: str = "medium",
	sentiment: str = "neutral",
) -> IntentResult:
	"""Build a common valid mocked intent response."""
	return IntentResult(
		intent=intent,
		category="order",
		priority=priority,
		sentiment=sentiment,
		extracted_entities=[],
	)


def _article(
	article_id: str = "wrong_item_workflow",
	workflow_id: str = "wrong_item",
	ranking_score: float = 0.8,
) -> dict[str, Any]:
	"""Create a mock search result with all expected KB metadata."""
	return {
		"article_id": article_id,
		"title": "Wrong item workflow",
		"category": "orders",
		"reliability_score": 0.95,
		"workflow_id": workflow_id,
		"content": "Verify ordered and delivered product SKUs.",
		"similarity": ranking_score / 0.95,
		"ranking_score": ranking_score,
	}


def test_detect_intent_uses_mock_and_extracts_order_id(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Use the Pydantic LLM result while extracting explicit order IDs in code."""
	mock_call = lambda *_args: _intent_result()
	monkeypatch.setattr(intent_agent, "call_llm_json", mock_call)

	result = intent_agent.detect_intent("Please check ORD10009 for me")

	assert result.intent == "order_status"
	assert result.extracted_entities[0].name == "order_id"
	assert result.extracted_entities[0].value == "ORD10009"


def test_intent_keyword_fallback_handles_hinglish_payment_issue(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Classify a Hinglish failed-and-deducted payment without an LLM result."""
	monkeypatch.setattr(
		intent_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock model unavailable")),
	)

	result = intent_agent.detect_intent("ORD10002 payment nahi hua but paisa kata")

	assert result.intent == "payment_failure"
	assert result.priority == "high"
	assert result.extracted_entities[0].value == "ORD10002"


def test_intent_keyword_fallback_handles_typos_and_order_cancellation(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Classify a misspelled cancellation request deterministically."""
	monkeypatch.setattr(
		intent_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(ValueError("mock invalid output")),
	)

	result = intent_agent.detect_intent("Please cencel ORD10006")

	assert result.intent == "cancel_order"
	assert result.extracted_entities[0].value == "ORD10006"


def test_intent_keyword_fallback_marks_locked_account_high_priority(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Treat a locked-account report as a high-priority login issue."""
	monkeypatch.setattr(
		intent_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock failure")),
	)

	result = intent_agent.detect_intent("My account is locked and I cannot sign in")

	assert result.intent == "login_issue"
	assert result.priority == "high"


def test_intent_keyword_fallback_detects_angry_sentiment(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Mark clear anger without inventing an unsupported support intent."""
	monkeypatch.setattr(
		intent_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock failure")),
	)

	result = intent_agent.detect_intent("I am furious about this unacceptable service")

	assert result.intent == "general_query"
	assert result.sentiment == "angry"


def test_high_risk_intent_priority_is_enforced_after_llm(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Do not allow a low model priority to downgrade a payment risk."""
	monkeypatch.setattr(
		intent_agent,
		"call_llm_json",
		lambda *_args: _intent_result("refund_request", "low"),
	)

	result = intent_agent.detect_intent("I want a refund for ORD10002")

	assert result.priority == "high"


@pytest.mark.parametrize("customer_id", [f"C{number:03d}" for number in range(1, 11)])
def test_context_builds_from_each_seeded_customer(
	customer_id: str,
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Build database-grounded contexts for all ten required demo customers."""
	monkeypatch.setattr(
		context_agent,
		"call_llm_json",
		lambda *_args: ContextSummary(summary="Mocked database facts summary."),
	)

	context = context_agent.build_context(customer_id, session=seeded_session)

	assert context.profile.customer_id == customer_id
	assert context.tier in {"standard", "premium"}
	assert context.summary == "Mocked database facts summary."


def test_context_uses_seeded_ticket_order_and_subscription_facts(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Expose C001's ticket count and C004's actual subscription state."""
	monkeypatch.setattr(
		context_agent,
		"call_llm_json",
		lambda *_args: ContextSummary(summary="Known records."),
	)

	wrong_item_context = context_agent.build_context("C001", seeded_session)
	subscription_context = context_agent.build_context("C004", seeded_session)

	assert wrong_item_context.tier == "premium"
	assert wrong_item_context.profile.tier == "premium"
	assert wrong_item_context.past_tickets_count == 3
	assert wrong_item_context.recent_orders[0].order_id == "ORD10001"
	assert subscription_context.subscription.status == "paid_not_activated_sync_failed"


def test_context_summary_has_database_grounded_template_fallback(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Retain DB facts when the one-paragraph LLM summary fails."""
	monkeypatch.setattr(
		context_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock summary failure")),
	)

	context = context_agent.build_context("C003", seeded_session)

	assert context.profile.account_status == "locked"
	assert "Morgan Chen" in context.summary
	assert "locked" in context.summary


def test_context_summary_calls_llm_only_for_summary(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Pass database-derived facts to one mocked summary call."""
	calls: list[tuple[object, ...]] = []

	def mock_summary(*args: object) -> ContextSummary:
		calls.append(args)
		return ContextSummary(summary="Mock summary.")

	monkeypatch.setattr(context_agent, "call_llm_json", mock_summary)
	context_agent.build_context("C010", seeded_session)

	assert len(calls) == 1
	assert calls[0][2] is ContextSummary
	assert "ORD10009" in str(calls[0][1])


def test_knowledge_selects_a_candidate_workflow(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Select one workflow from the retrieved article candidates."""
	monkeypatch.setattr(knowledge_agent, "search_kb", lambda *_args, **_kwargs: [_article()])
	monkeypatch.setattr(
		knowledge_agent,
		"call_llm_json",
		lambda *_args: knowledge_agent.WorkflowSelection(
			workflow_id="wrong_item", confidence=0.9, reasoning="SKU mismatch workflow."
		),
	)

	result = knowledge_agent.select_workflow("wrong_item", None, "I got the wrong item")

	assert result.selected_workflow == "wrong_item"
	assert result.confidence == 0.9
	assert result.articles[0].article_id == "wrong_item_workflow"


def test_knowledge_uses_intent_workflow_when_retrieval_misses_it(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Keep a payment failure on its canonical path when retrieval returns unrelated hits."""
	monkeypatch.setattr(
		knowledge_agent,
		"search_kb",
		lambda *_args, **_kwargs: [_article("cancel", "cancel_order", 0.95)],
	)
	monkeypatch.setattr(
		knowledge_agent,
		"call_llm_json",
		lambda *_args: pytest.fail("LLM should not select an incompatible workflow"),
	)
	result = knowledge_agent.select_workflow("payment_failure", None, "payment failed")

	assert result.selected_workflow == "payment_deducted"
	assert result.confidence == 1.0


def test_knowledge_returns_none_for_low_confidence(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Do not select a workflow when Gemini confidence is below 0.5."""
	monkeypatch.setattr(knowledge_agent, "search_kb", lambda *_args, **_kwargs: [_article()])
	monkeypatch.setattr(
		knowledge_agent,
		"call_llm_json",
		lambda *_args: knowledge_agent.WorkflowSelection(
			workflow_id="wrong_item", confidence=0.49, reasoning="Weak match."
		),
	)

	result = knowledge_agent.select_workflow("wrong_item", None, "question")

	assert result.selected_workflow == "none"


def test_knowledge_returns_none_without_articles(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Avoid an LLM call when retrieval returns no candidate articles."""
	monkeypatch.setattr(knowledge_agent, "search_kb", lambda *_args, **_kwargs: [])
	monkeypatch.setattr(
		knowledge_agent,
		"call_llm_json",
		lambda *_args: pytest.fail("LLM should not be called without candidates"),
	)

	result = knowledge_agent.select_workflow("general_query", None, "unknown topic")

	assert result.selected_workflow == "none"
	assert result.confidence == 0.0


def test_knowledge_rejects_workflow_not_in_candidates(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Replace an unsupported LLM workflow with the strongest retrieved candidate."""
	monkeypatch.setattr(knowledge_agent, "search_kb", lambda *_args, **_kwargs: [_article()])
	monkeypatch.setattr(
		knowledge_agent,
		"call_llm_json",
		lambda *_args: knowledge_agent.WorkflowSelection(
			workflow_id="invented_workflow", confidence=0.9, reasoning="Not retrieved."
		),
	)

	result = knowledge_agent.select_workflow("wrong_item", None, "wrong product")

	assert result.selected_workflow == "wrong_item"
	assert result.confidence == 0.8


def test_knowledge_uses_ranked_article_fallback_on_llm_error(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Choose the best reliable candidate if workflow selection fails."""
	monkeypatch.setattr(
		knowledge_agent,
		"search_kb",
		lambda *_args, **_kwargs: [
			_article("weaker", "general_info", 0.55),
			_article("stronger", "wrong_item", 0.86),
		],
	)
	monkeypatch.setattr(
		knowledge_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock LLM down")),
	)

	result = knowledge_agent.select_workflow("wrong_item", None, "wrong item")

	assert result.selected_workflow == "wrong_item"
	assert result.confidence == 0.86


def _mock_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Replace Gemini diagnosis phrasing with a deterministic test response."""
	monkeypatch.setattr(
		troubleshooting_agent,
		"call_llm_json",
		lambda *_args: DiagnosisResponse(diagnosis="Mocked diagnosis from verified facts."),
	)


def test_wrong_item_runner_verifies_c001_sku_mismatch(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Use order, item, and shipment tools before proposing replacement."""
	_mock_diagnosis(monkeypatch)

	result = troubleshooting_agent.troubleshoot(
		"wrong_item", {"order_id": "ORD10001"}, "wrong_item", seeded_session
	)

	assert [step.step for step in result.steps_run] == [
		"get_order",
		"get_order_items",
		"get_shipment",
	]
	assert result.proposed_action == "schedule_replacement"


def test_cancel_runner_blocks_shipped_c005(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Propose escalation for C005's already-shipped order."""
	_mock_diagnosis(monkeypatch)
	result = troubleshooting_agent.troubleshoot(
		"cancel_order", {"order_id": "ORD10005"}, "cancel_order", seeded_session
	)
	assert result.proposed_action == "escalate_policy_block"
	assert "shipped" in result.steps_run[0].finding


def test_cancel_runner_allows_placed_c006(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Propose cancellation only for C006's placed order."""
	_mock_diagnosis(monkeypatch)
	result = troubleshooting_agent.troubleshoot(
		"cancel_order", {"order_id": "ORD10006"}, "cancel_order", seeded_session
	)
	assert result.proposed_action == "cancel_order"


def test_payment_runner_detects_c002_deduction(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Use payment logs to detect C002's failed-but-deducted transaction."""
	_mock_diagnosis(monkeypatch)
	result = troubleshooting_agent.troubleshoot(
		"payment_failure", {"order_id": "ORD10002"}, "payment_deducted", seeded_session
	)
	assert result.proposed_action == "initiate_refund"
	assert result.steps_run[-1].step == "get_payment_logs"


def test_account_runner_detects_locked_c003(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Propose unlock for C003 only after reading its locked status."""
	_mock_diagnosis(monkeypatch)
	result = troubleshooting_agent.troubleshoot(
		"login_issue", {"customer_id": "C003"}, "account_locked", seeded_session
	)
	assert result.proposed_action == "unlock_account"


def test_subscription_runner_detects_c004_sync_failure(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Escalate C004's seeded paid-but-not-activated subscription state."""
	_mock_diagnosis(monkeypatch)
	result = troubleshooting_agent.troubleshoot(
		"subscription_issue", {"customer_id": "C004"}, "subscription_sync", seeded_session
	)
	assert result.proposed_action == "escalate_subscription_sync"


def test_order_status_runner_uses_c010_order(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Read C010's order and shipment before proposing a status reply."""
	_mock_diagnosis(monkeypatch)
	result = troubleshooting_agent.troubleshoot(
		"order_status", {"order_id": "ORD10009"}, "order_status", seeded_session
	)
	assert result.proposed_action == "share_verified_order_status"
	assert len(result.steps_run) == 2


def test_troubleshooting_falls_back_to_template_if_diagnosis_fails(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Keep deterministic findings and action when diagnosis phrasing fails."""
	monkeypatch.setattr(
		troubleshooting_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock diagnosis error")),
	)
	result = troubleshooting_agent.troubleshoot(
		"cancel_order", {"order_id": "ORD10005"}, "cancel_order", seeded_session
	)
	assert result.proposed_action == "escalate_policy_block"
	assert "support specialist" in result.diagnosis


def test_agent_response_schemas_pass_gemini_compatibility_check() -> None:
	"""Validate all public response schemas against the Gemini client rules."""
	from llm.client import _validate_response_schema

	for schema in (
		IntentResult,
		CustomerContext,
		KBResult,
		troubleshooting_agent.TroubleshootResult,
		ActionResult,
		ConversationState,
		EscalationDecision,
		Ticket,
		CSATResult,
		ContextSummary,
		DiagnosisResponse,
		knowledge_agent.WorkflowSelection,
	):
		_validate_response_schema(schema)