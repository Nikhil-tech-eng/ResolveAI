"""Tests for ticket tracking and CSAT agents with mocked sentiment calls."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from agents import csat_agent, tracking_agent
from agents.csat_agent import CommentSentiment
from agents.schemas import CSATResult
from data.seed_data import seed_database
from db.models import ActionLog, CSAT, Customer, Ticket as DatabaseTicket


@pytest.fixture
def seeded_session(tmp_path: Path) -> Iterator[Session]:
	"""Create an isolated database with the standard seeded support scenarios."""
	database_path = tmp_path / "support.db"
	database_url = f"sqlite:///{database_path.as_posix()}"
	seed_database(database_url)
	engine = create_engine(database_url, connect_args={"check_same_thread": False})
	with Session(engine) as session:
		yield session
	engine.dispose()


def _new_ticket(
	session: Session,
	customer_id: str,
	*,
	status: str = "opened",
	sla_deadline: datetime | None = None,
) -> DatabaseTicket:
	"""Create and persist a ticket for one seeded customer."""
	ticket = DatabaseTicket(
		customer_id=customer_id,
		intent="general_query",
		category="general",
		priority="medium",
		sentiment="neutral",
		summary="Test ticket",
		investigation="Test investigation",
		actions_taken=[],
		status=status,
		sla_deadline=sla_deadline,
	)
	session.add(ticket)
	session.commit()
	return ticket


def test_valid_ticket_state_machine_transitions_notify_each_time(
	seeded_session: Session,
) -> None:
	"""Exercise the normal flow, reassignment, and resolved-ticket reopen path."""
	ticket = seeded_session.scalar(
		select(DatabaseTicket).where(DatabaseTicket.customer_id == "C007")
	)
	assert ticket is not None
	transitions = [
		"assigned",
		"pending_customer",
		"assigned",
		"pending_customer",
		"resolved",
		"opened",
		"assigned",
		"pending_customer",
		"resolved",
		"closed",
	]
	for next_status in transitions:
		result = tracking_agent.transition(ticket.id, next_status, "agent-1", seeded_session)
		assert result["new_status"] == next_status
		assert result["notification"]["success"] is True

	assert ticket.status == "closed"
	assert seeded_session.scalar(
		select(func.count(ActionLog.id)).where(ActionLog.tool_name == "send_notification")
	) == len(transitions)


def test_invalid_transition_keeps_ticket_unchanged_and_unnotified(
	seeded_session: Session,
) -> None:
	"""Reject skipping directly from opened to closed without a notification."""
	ticket = _new_ticket(seeded_session, "C010")
	with pytest.raises(ValueError, match="Invalid ticket transition: opened -> closed"):
		tracking_agent.transition(ticket.id, "closed", "agent-2", seeded_session)

	seeded_session.refresh(ticket)
	assert ticket.status == "opened"
	assert seeded_session.scalar(
		select(func.count(ActionLog.id)).where(ActionLog.tool_name == "send_notification")
	) == 0


def test_closed_ticket_cannot_be_reassigned(seeded_session: Session) -> None:
	"""Reject any transition out of the terminal closed state."""
	ticket = _new_ticket(seeded_session, "C010", status="closed")
	with pytest.raises(ValueError, match="Allowed next status: none"):
		tracking_agent.transition(ticket.id, "assigned", "agent-2", seeded_session)


def test_transition_rejects_unknown_ticket(seeded_session: Session) -> None:
	"""Return a clear error when the requested ticket does not exist."""
	with pytest.raises(ValueError, match="Ticket 999999 was not found"):
		tracking_agent.transition(999999, "assigned", "agent-2", seeded_session)


def test_sla_check_flags_only_tickets_inside_twenty_percent_window(
	seeded_session: Session,
) -> None:
	"""Mark near-deadline premium and standard tickets high priority."""
	now = datetime.now(timezone.utc)
	premium_ticket = _new_ticket(
		seeded_session, "C001", sla_deadline=now + timedelta(minutes=2)
	)
	standard_ticket = _new_ticket(
		seeded_session, "C002", sla_deadline=now + timedelta(minutes=11)
	)
	far_ticket = _new_ticket(
		seeded_session, "C010", sla_deadline=now + timedelta(minutes=30)
	)

	flagged = tracking_agent.sla_check(seeded_session, now=now)
	flagged_ids = {item["ticket_id"] for item in flagged}

	assert flagged_ids == {premium_ticket.id, standard_ticket.id}
	assert premium_ticket.priority == standard_ticket.priority == "high"
	assert premium_ticket.sla_risk and standard_ticket.sla_risk
	assert not far_ticket.sla_risk
	assert tracking_agent.sla_check(seeded_session, now=now) == []


def _mock_sentiment(monkeypatch: pytest.MonkeyPatch, sentiment: str) -> None:
	"""Mock the single structured sentiment response for a comment."""
	monkeypatch.setattr(
		csat_agent,
		"call_llm_json",
		lambda *_args: CommentSentiment(sentiment=sentiment),
	)


def test_collect_persists_full_resolution_feedback_for_c004(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Store C004 feedback, link its ticket, and classify a resolved conversation."""
	conversation = seeded_session.get(csat_agent.Conversation, "CONV-C004")
	assert conversation is not None
	ticket = _new_ticket(seeded_session, "C004", status="resolved")
	conversation.state = {
		**conversation.state,
		"ticket_id": ticket.id,
		"memory_state": {"lifecycle_stage": "resolved"},
	}
	seeded_session.commit()
	_mock_sentiment(monkeypatch, "positive")

	result = csat_agent.collect(
		"CONV-C004", 5, "Great and helpful support, thanks!", seeded_session
	)
	stored = seeded_session.scalar(
		select(CSAT).where(CSAT.conversation_id == "CONV-C004")
	)

	assert result == CSATResult(
		rating=5,
		sentiment="positive",
		resolution_quality="resolved_fully",
	)
	assert stored is not None and stored.ticket_id == ticket.id
	assert stored.comment == "Great and helpful support, thanks!"


def test_collect_uses_keyword_fallback_and_not_resolved_quality(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Fall back to deterministic angry sentiment for unresolved C008 feedback."""
	monkeypatch.setattr(
		csat_agent,
		"call_llm_json",
		lambda *_args: (_ for _ in ()).throw(RuntimeError("mock Gemini failure")),
	)

	result = csat_agent.collect(
		"CONV-C008", 1, "I am furious; this was unacceptable.", seeded_session
	)

	assert result.sentiment == "angry"
	assert result.resolution_quality == "not_resolved"


def test_collect_marks_action_without_resolution_as_partial(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Classify a conversation with successful actions but no resolved state as partial."""
	_mock_sentiment(monkeypatch, "neutral")

	result = csat_agent.collect("CONV-C006", 3, "It helped somewhat.", seeded_session)

	assert result.resolution_quality == "partially"


def test_collect_rejects_rating_outside_one_to_five_without_llm(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Validate the rating before spending an LLM call on comment sentiment."""
	monkeypatch.setattr(
		csat_agent,
		"call_llm_json",
		lambda *_args: pytest.fail("Invalid ratings must be rejected before sentiment analysis"),
	)
	with pytest.raises(ValidationError):
		csat_agent.collect("CONV-C010", 6, "Fine", seeded_session)


def test_collect_rejects_missing_conversation(seeded_session: Session) -> None:
	"""Avoid persisting feedback for an unknown conversation ID."""
	with pytest.raises(ValueError, match="Conversation CONV-MISSING was not found"):
		csat_agent.collect("CONV-MISSING", 4, "Okay", seeded_session)


def test_csat_summary_returns_rating_and_quality_aggregates(
	seeded_session: Session,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Aggregate saved ratings, sentiment labels, and resolution quality."""
	seeded_session.execute(delete(CSAT))
	seeded_session.commit()
	_mock_sentiment(monkeypatch, "positive")
	csat_agent.collect("CONV-C010", 5, "Great", seeded_session)
	csat_agent.collect("CONV-C009", 3, "Helpful", seeded_session)

	result = csat_agent.csat_summary(seeded_session)

	assert result["count"] == 2
	assert result["average_rating"] == 4.0
	assert result["sentiment_distribution"] == {"positive": 2}
	assert result["resolution_quality_distribution"] == {"not_resolved": 2}