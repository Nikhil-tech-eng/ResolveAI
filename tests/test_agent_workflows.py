"""Tests for action, memory, escalation, and ticket agents with mocked Gemini."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agents import action_agent, escalation_agent, memory_agent, ticket_agent
from agents.schemas import ActionMessage, EscalationDecision, TicketText
from data.seed_data import seed_database
from db.models import ActionLog, Base, Conversation, Ticket as DatabaseTicket


@pytest.fixture
def seeded_session(tmp_path: Path) -> Iterator[Session]:
	"""Create a fresh seeded SQLite database for one workflow test."""
	database_path = tmp_path / "support.db"
	database_url = f"sqlite:///{database_path.as_posix()}"
	seed_database(database_url)
	engine = create_engine(database_url, connect_args={"check_same_thread": False})
	with Session(engine) as session:
		yield session
	engine.dispose()


def _mock_action_message(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Replace customer-message phrasing with a fixed schema-valid response."""
	monkeypatch.setattr(
		action_agent,
		"call_llm_json",
		lambda *_args: ActionMessage(customer_message="The tool results have been reviewed."),
	)


def test_replace_and_return_runs_all_steps_and_notifies(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Run return, label, verified replacement, and notification for C001."""
	_mock_action_message(monkeypatch)

	result = action_agent.execute(
		"replace_and_return",
		{"order_id": "ORD10001"},
		"C001",
		"CONV-C001",
		session=seeded_session,
	)

	assert result.success is True
	assert [action.tool_name for action in result.actions_executed] == [
		"create_return_request",
		"generate_return_label",
		"schedule_replacement",
		"send_notification",
	]
	assert result.customer_message == "The tool results have been reviewed."


def test_cancel_stops_after_policy_blocks_shipped_c005(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Stop a cancellation sequence and report the policy result for C005."""
	_mock_action_message(monkeypatch)

	result = action_agent.execute(
		"cancel",
		{"order_id": "ORD10005"},
		"C005",
		"CONV-C005",
		session=seeded_session,
	)

	assert result.success is False
	assert len(result.actions_executed) == 1
	assert result.actions_executed[0].success is False
	assert "only orders with status 'placed'" in result.actions_executed[0].summary
	assert not any(action.tool_name == "send_notification" for action in result.actions_executed)


def test_refund_runs_and_notifies_using_c002_payment_facts(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Refund the failed-but-deducted C002 payment and notify the customer."""
	_mock_action_message(monkeypatch)

	result = action_agent.execute(
		"refund",
		{"order_id": "ORD10002"},
		"C002",
		"CONV-C002",
		session=seeded_session,
	)

	assert result.success is True
	assert [action.tool_name for action in result.actions_executed] == [
		"initiate_refund",
		"send_notification",
	]


def test_unlock_and_reset_runs_sequentially_for_c003(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Unlock C003 before resetting its password and sending notification."""
	_mock_action_message(monkeypatch)

	result = action_agent.execute(
		"unlock_and_reset",
		{},
		"C003",
		"CONV-C003",
		session=seeded_session,
	)

	assert result.success is True
	assert [action.tool_name for action in result.actions_executed] == [
		"unlock_account",
		"reset_password",
		"send_notification",
	]


def test_return_sequence_stops_for_c007_policy_block(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Do not generate labels or replacements after C007's return is blocked."""
	_mock_action_message(monkeypatch)

	result = action_agent.execute(
		"replace_and_return",
		{"order_id": "ORD10007"},
		"C007",
		"CONV-C007",
		session=seeded_session,
	)

	assert result.success is False
	assert [action.tool_name for action in result.actions_executed] == [
		"create_return_request"
	]
	assert "30 days" in result.actions_executed[0].summary


def test_memory_persists_entities_across_three_turns(
	seeded_session: Session,
) -> None:
	"""Track a missing order ID, resolve it, and persist through three turns."""
	conversation = Conversation(
		id="CONV-MEMORY-THREE-TURNS",
		customer_id="C010",
		channel="web",
		state={},
		status="open",
	)
	seeded_session.add(conversation)
	seeded_session.commit()

	initial = memory_agent.load_state(
		conversation.id, intent="order_status", session=seeded_session
	)
	assert initial.missing_fields == ["order_id"]
	assert memory_agent.next_question(initial) == "What is your order ID? It usually starts with ORD."

	turn_one = memory_agent.update_state(
		conversation.id,
		"order_status",
		entities={"channel": "web"},
		session=seeded_session,
	)
	turn_two = memory_agent.update_state(
		conversation.id,
		"order_status",
		entities={"order_id": "ORD10009"},
		lifecycle_stage="investigating",
		session=seeded_session,
	)
	turn_three = memory_agent.update_state(
		conversation.id,
		"order_status",
		entities={"verified_status": "shipped"},
		lifecycle_stage="resolved",
		session=seeded_session,
	)
	loaded = memory_agent.load_state(conversation.id, session=seeded_session)

	assert turn_one.turn_count == 1
	assert turn_one.lifecycle_stage == "gathering_info"
	assert turn_two.turn_count == 2 and not turn_two.missing_fields
	assert turn_three.turn_count == loaded.turn_count == 3
	assert loaded.lifecycle_stage == "resolved"
	assert {entity.name: entity.value for entity in loaded.entities}["order_id"] == "ORD10009"
	assert {entity.name: entity.value for entity in loaded.entities}["verified_status"] == "shipped"


NOW = datetime.now(timezone.utc)
HARD_ESCALATION_CASES = [
	pytest.param({"intent": "fraud_complaint", "confidence": 0.95}, id="fraud"),
	pytest.param({"intent": "legal_request", "confidence": 0.95}, id="legal"),
	pytest.param({"intent": "data_breach", "confidence": 0.95}, id="data-breach"),
	pytest.param({"intent": "order_status", "confidence": 0.69}, id="low-confidence"),
	pytest.param({"intent": "order_status", "sentiment": "angry", "attempts": 1}, id="angry-after-one"),
	pytest.param({"intent": "order_status", "sentiment": "negative", "attempts": 2}, id="negative-after-two"),
	pytest.param({"intent": "cancel_order", "policy_blocked": True, "customer_disputes": True}, id="disputed-policy-block"),
	pytest.param({"intent": "order_status", "sla_at_risk": True}, id="sla-flag"),
	pytest.param({"intent": "order_status", "sla_deadline": (NOW - timedelta(seconds=1)).isoformat()}, id="sla-deadline"),
	pytest.param({"intent": "order_status", "tier": "premium", "sla_started_at": (NOW - timedelta(minutes=16)).isoformat()}, id="premium-sla"),
	pytest.param({"intent": "order_status", "tier": "standard", "sla_started_at": (NOW - timedelta(minutes=61)).isoformat()}, id="standard-sla"),
	pytest.param({"intent": "order_status", "steps_run": [{"finding": "error"}, {"finding": "failed"}, {"finding": "not found"}]}, id="three-failed-steps"),
	pytest.param({"intent": "order_status", "failed_steps": 3}, id="failed-step-count"),
]


@pytest.mark.parametrize("state", HARD_ESCALATION_CASES)
def test_hard_escalation_rules_do_not_call_llm(
	state: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Verify each mandatory escalation rule wins before borderline evaluation."""
	monkeypatch.setattr(
		escalation_agent,
		"call_llm_json",
		lambda *_args: pytest.fail("Hard rules must not call the LLM"),
	)

	decision = escalation_agent.decide(state)

	assert decision.decision == "escalate"
	assert decision.reason.strip()


def test_borderline_escalation_uses_mocked_llm() -> None:
	"""Use Gemini only when no hard escalation rule has matched."""
	escalation_agent.call_llm_json = lambda *_args: EscalationDecision(
		decision="auto_resolve", reason="Verified routine order update.", confidence=0.91
	)

	decision = escalation_agent.decide(
		{"intent": "order_status", "confidence": 0.91, "resolved": True}
	)

	assert decision.decision == "auto_resolve"
	assert decision.reason == "Verified routine order update."


def test_borderline_llm_failure_falls_back_to_escalation(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Escalate safely with a reason when borderline evaluation fails."""
	monkeypatch.setattr(
		escalation_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock outage")),
	)

	decision = escalation_agent.decide(
		{"intent": "order_status", "confidence": 0.91, "resolved": True}
	)

	assert decision.decision == "escalate"
	assert "human review" in decision.reason


def _ticket_state(customer_id: str, conversation_id: str, intent: str) -> dict[str, object]:
	"""Build graph state for a seeded scenario ticket."""
	return {
		"customer_id": customer_id,
		"conversation_id": conversation_id,
		"intent": {
			"intent": intent,
			"category": "subscription" if intent == "subscription_issue" else "safety",
			"priority": "high" if intent == "fraud_complaint" else "medium",
			"sentiment": "negative" if intent == "fraud_complaint" else "neutral",
			"extracted_entities": [],
		},
		"context": {"tier": "premium" if customer_id == "C004" else "standard"},
		"messages": [
			{
				"role": "customer",
				"content": "My subscription was paid but did not activate."
				if customer_id == "C004"
				else "I see charges I do not recognize.",
			}
		],
		"troubleshoot": {
			"steps_run": [
				{"step": "read_customer", "finding": "Customer record confirmed."},
				{"step": "check_payment", "finding": "No transaction has been confirmed by the customer."},
			],
			"proposed_action": "escalate",
		},
		"escalation": {
			"decision": "escalate",
			"reason": "Specialist review is required.",
			"confidence": 0.99,
		},
	}


@pytest.mark.parametrize(
	("customer_id", "conversation_id", "intent", "tier", "expected_minutes"),
	[
		("C004", "CONV-C004", "subscription_issue", "premium", 15),
		("C008", "CONV-C008", "fraud_complaint", "standard", 60),
	],
)
def test_ticket_for_seeded_scenarios_persists_actions_and_sla(
	customer_id: str,
	conversation_id: str,
	intent: str,
	tier: str,
	expected_minutes: int,
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Persist human-ready C004/C008 tickets with tier-correct SLA deadlines."""
	monkeypatch.setattr(
		ticket_agent,
		"call_llm_json",
		lambda *_args: TicketText(
			issue_summary="The customer reports a verified support concern.",
			suggested_next_step="Review the recorded evidence and contact the customer.",
		),
	)
	state = _ticket_state(customer_id, conversation_id, intent)

	ticket = ticket_agent.generate_ticket(state, session=seeded_session)
	row = seeded_session.get(DatabaseTicket, ticket.ticket_id)

	assert row is not None
	assert ticket.customer_id == customer_id
	assert ticket.customer_tier == tier
	assert ticket.status == "opened"
	assert ticket.sla_deadline is not None
	assert ticket.assigned_to == "support_queue"
	assert ticket.summary == "The customer reports a verified support concern."
	assert ticket.suggested_next_step.startswith("Review the recorded evidence")
	assert row.escalation_reason == "Specialist review is required."
	assert row.investigation.startswith("read_customer:")
	assert row.sla_deadline is not None
	assert timedelta(minutes=expected_minutes - 1) < row.sla_deadline - row.created_at < timedelta(minutes=expected_minutes + 1)