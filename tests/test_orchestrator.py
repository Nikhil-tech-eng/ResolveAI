"""Mocked end-to-end orchestrator tests for the five required demo scenarios."""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from agents import (
	action_agent,
	context_agent,
	csat_agent,
	escalation_agent,
	intent_agent,
	knowledge_agent,
	ticket_agent,
	troubleshooting_agent,
)
from agents.csat_agent import CommentSentiment
from agents.orchestrator import handle_message
from agents.schemas import (
	ActionMessage,
	ContextSummary,
	DiagnosisResponse,
	EscalationDecision,
	IntentResult,
	TicketText,
	WorkflowSelection,
)
from data.seed_data import seed_database
from db.models import ActionLog, Base, Conversation, CSAT, Message, Order, Ticket


@pytest.fixture
def seeded_session(tmp_path: Path) -> Iterator[Session]:
	"""Create an isolated database containing customers C001 through C010."""
	database_path = tmp_path / "support.db"
	database_url = f"sqlite:///{database_path.as_posix()}"
	seed_database(database_url)
	engine = create_engine(database_url, connect_args={"check_same_thread": False})
	with Session(engine) as session:
		yield session
	engine.dispose()


def _mock_model_response(
	system: str,
	user: str,
	schema: type[Any],
	agent: str = "default",
) -> Any:
	"""Return deterministic Pydantic values for every model call in this graph."""
	if schema is IntentResult:
		text = user.lower()
		if "fraud" in text or "unauthorized" in text or "unknown charge" in text:
			intent = "fraud_complaint"
			category = "safety"
		elif "payment" in text or "deducted" in text:
			intent = "payment_failure"
			category = "payment"
		elif "wrong item" in text:
			intent = "wrong_item"
			category = "order"
		elif "cancel" in text:
			intent = "cancel_order"
			category = "order"
		else:
			intent = "order_status"
			category = "order"
		sentiment = "angry" if any(word in text for word in ("furious", "angry", "mad")) else "neutral"
		priority = "high" if intent in {"payment_failure", "fraud_complaint"} else "medium"
		entities = [
			{"name": "order_id", "value": order_id.upper()}
			for order_id in re.findall(r"\bORD\d+\b", user, re.IGNORECASE)
		]
		return IntentResult(
			intent=intent,
			category=category,
			priority=priority,
			sentiment=sentiment,
			extracted_entities=entities,
		)
	if schema is ContextSummary:
		return ContextSummary(summary="Seeded database context loaded.")
	if schema is WorkflowSelection:
		request = json.loads(user)
		workflow = request["candidates"][0]["workflow_id"]
		return WorkflowSelection(workflow_id=workflow, confidence=0.95, reasoning="Mocked candidate selection.")
	if schema is DiagnosisResponse:
		return DiagnosisResponse(diagnosis="I checked the available account and order records.")
	if schema is EscalationDecision:
		return EscalationDecision(
			decision="auto_resolve", reason="Mocked routine case decision.", confidence=0.95
		)
	if schema is ActionMessage:
		results = json.loads(user)
		failure = next(
			(entry["result"].get("error") for entry in results if not entry["result"]["success"]),
			None,
		)
		message = f"I couldn't complete that action: {failure}" if failure else "Your support action is complete."
		return ActionMessage(customer_message=message)
	if schema is TicketText:
		return TicketText(
			issue_summary="The customer requires support follow-up.",
			suggested_next_step="Review the verified findings and contact the customer.",
		)
	if schema is CommentSentiment:
		return CommentSentiment(sentiment="positive" if "helpful" in user.lower() else "neutral")
	raise AssertionError(f"Unexpected mocked response schema: {schema.__name__}")


def _install_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Mock every agent LLM boundary and the knowledge retrieval result."""
	for module in (
		intent_agent,
		context_agent,
		knowledge_agent,
		troubleshooting_agent,
		escalation_agent,
		action_agent,
		ticket_agent,
		csat_agent,
	):
		monkeypatch.setattr(module, "call_llm_json", _mock_model_response)

	def mock_search(query: str, category: str | None = None, k: int = 4) -> list[dict[str, Any]]:
		"""Return one candidate article for the intent in the orchestrator query."""
		query_text = query.lower()
		if "wrong_item" in query_text:
			workflow = "wrong_item"
		elif "payment_failure" in query_text:
			workflow = "payment_deducted"
		elif "fraud_complaint" in query_text:
			workflow = "fraud_escalate"
		elif "cancel_order" in query_text:
			workflow = "cancel_order"
		else:
			workflow = "order_status"
		return [
			{
				"article_id": f"mock-{workflow}",
				"title": f"Mock {workflow} procedure",
				"category": "support",
				"content": "Verified mock workflow article.",
				"reliability_score": 0.95,
				"workflow_id": workflow,
				"similarity": 1.0,
				"ranking_score": 0.95,
			}
		]

	monkeypatch.setattr(knowledge_agent, "search_kb", mock_search)


def test_c001_wrong_item_auto_resolves_in_one_turn(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Verify wrong-item facts lead to replacement action and a one-turn reply."""
	_install_mocks(monkeypatch)
	result = handle_message(
		"CONV-C001",
		"C001",
		"I got the wrong item for ORD10001.",
		"web",
		seeded_session,
	)

	assert result["state"]["escalation"].decision == "auto_resolve"
	assert result["state"]["actions"].success is True
	assert "1 to 5" in result["reply"]
	assert any(entry["agent"] == "action" for entry in result["trace"])
	assert all(
		set(entry)
		== {"agent", "input_summary", "output_summary", "tools_called", "duration_ms", "confidence"}
		for entry in result["trace"]
	)
	assert seeded_session.scalar(
		select(func.count(Message.id)).where(
			Message.conversation_id == "CONV-C001", Message.role == "customer"
		)
	) >= 1


def test_c002_payment_failure_runs_three_turns_and_collects_csat(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Ask for the missing order ID, refund on turn two, and store rating on turn three."""
	_install_mocks(monkeypatch)
	first = handle_message(
		"CONV-C002-THREE-TURNS",
		"C002",
		"My payment failed and the amount was deducted.",
		"web",
		seeded_session,
	)
	second = handle_message(
		"CONV-C002-THREE-TURNS",
		"C002",
		"The order is ORD10002.",
		"web",
		seeded_session,
	)
	third = handle_message(
		"CONV-C002-THREE-TURNS",
		"C002",
		"5, this was helpful.",
		"web",
		seeded_session,
	)

	order = seeded_session.get(Order, "ORD10002")
	conversation = seeded_session.get(Conversation, "CONV-C002-THREE-TURNS")
	assert "order ID" in first["reply"]
	assert second["state"]["actions"].success is True
	assert "1 to 5" in second["reply"]
	assert order is not None and order.payment_status == "refunded"
	assert third["state"]["csat"].rating == 5
	assert conversation is not None
	assert conversation.state["memory_state"]["turn_count"] == 3
	assert seeded_session.scalar(
		select(func.count(CSAT.id)).where(CSAT.conversation_id == "CONV-C002-THREE-TURNS")
	) == 1
	assert seeded_session.scalar(
		select(func.count(Message.id)).where(
			Message.conversation_id == "CONV-C002-THREE-TURNS",
			Message.role == "customer",
		)
	) == 3


def test_c008_fraud_escalates_and_creates_assigned_ticket(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Route the high-risk fraud complaint through ticketing and tracking."""
	_install_mocks(monkeypatch)
	result = handle_message(
		"CONV-C008",
		"C008",
		"I see an unauthorized charge and believe this is fraud.",
		"web",
		seeded_session,
	)
	ticket = result["ticket"]
	stored_ticket = seeded_session.get(Ticket, ticket.ticket_id)

	assert result["state"]["escalation"].decision == "escalate"
	assert ticket.intent == "fraud_complaint"
	assert ticket.status == "assigned"
	assert stored_ticket is not None and stored_ticket.assigned_to == "support_queue"
	assert any(entry["agent"] == "ticket" for entry in result["trace"])
	assert any(entry["agent"] == "tracking" for entry in result["trace"])


def test_c005_policy_block_is_explained_without_claiming_cancel(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Explain the shipped-order cancellation policy using the action result."""
	_install_mocks(monkeypatch)
	result = handle_message(
		"CONV-C005",
		"C005",
		"Please cancel ORD10005; it has already shipped.",
		"web",
		seeded_session,
	)
	order = seeded_session.get(Order, "ORD10005")

	assert result["state"]["actions"].success is False
	assert "shipped" in result["reply"].lower()
	assert "cannot cancel order" in result["reply"].lower()
	assert order is not None and order.status == "shipped"
	assert result["ticket"] is None


def test_angry_c009_gets_apology_then_escalates_after_one_failed_attempt(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Apologize on the first turn, then escalate with high ticket priority."""
	_install_mocks(monkeypatch)
	first = handle_message(
		"CONV-C009",
		"C009",
		"My ORD10008 order is still delayed. I am furious.",
		"web",
		seeded_session,
	)
	conversation = seeded_session.get(Conversation, "CONV-C009")
	assert first["reply"].startswith("I'm sorry")
	assert conversation is not None and conversation.state["failed_attempts"] == 1

	second = handle_message(
		"CONV-C009",
		"C009",
		"It is still not fixed.",
		"web",
		seeded_session,
	)

	assert second["reply"].startswith("I'm sorry")
	assert second["state"]["escalation"].decision == "escalate"
	assert second["ticket"].priority == "high"
	assert second["ticket"].sentiment == "angry"