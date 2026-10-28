"""Focused tests for root-cause alerts and SLA ticket flagging."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from analytics.root_cause import detect_root_cause_spikes
from agents.tracking_agent import sla_check
from db.models import Base, Conversation, Customer, Ticket


@pytest.fixture
def analytics_session(tmp_path: Path) -> Iterator[Session]:
	"""Create a minimal isolated support database for deterministic analytics."""
	database_path = tmp_path / "analytics.db"
	engine = create_engine(f"sqlite:///{database_path.as_posix()}")
	Base.metadata.create_all(engine)
	with Session(engine) as session:
		session.add_all(
			[
				Customer(id="C001", name="Premium Customer", email="premium@example.com", tier="premium"),
				Customer(id="C002", name="Standard Customer", email="standard@example.com", tier="standard"),
			]
		)
		session.commit()
		yield session
	engine.dispose()


def _add_conversation(
	session: Session,
	conversation_id: str,
	intent: str,
	started_at: datetime,
	customer_id: str = "C002",
) -> None:
	"""Insert one conversation carrying a top-level intent label."""
	session.add(
		Conversation(
			id=conversation_id,
			customer_id=customer_id,
			channel="test",
			state={"intent": intent},
			status="open",
			started_at=started_at,
		)
	)


def _add_ticket(
	session: Session,
	customer_id: str,
	*,
	status: str = "opened",
	sla_deadline: datetime,
) -> Ticket:
	"""Insert a ticket with the required database fields and SLA deadline."""
	ticket = Ticket(
		customer_id=customer_id,
		intent="payment_failure",
		category="payment",
		priority="medium",
		sentiment="neutral",
		summary="Payment support issue",
		investigation="Payment records reviewed",
		actions_taken=[],
		status=status,
		sla_deadline=sla_deadline,
	)
	session.add(ticket)
	session.flush()
	return ticket


def test_root_cause_alerts_payment_spike_over_ten(
	analytics_session: Session,
) -> None:
	"""Create a payment outage alert when the current window exceeds ten reports."""
	now = datetime.now(timezone.utc)
	for index in range(11):
		_add_conversation(
			analytics_session,
			f"PAY-{index}",
			"payment_failure",
			now - timedelta(minutes=20, seconds=index),
		)
	analytics_session.commit()

	alerts = detect_root_cause_spikes(analytics_session, now=now)

	assert len(alerts) == 1
	assert alerts[0].intent == "payment_failure"
	assert alerts[0].count == 11
	assert "possible payment gateway outage" in alerts[0].message


def test_root_cause_alerts_three_times_baseline_below_absolute_threshold(
	analytics_session: Session,
) -> None:
	"""Detect a seven-report spike against a two-report half-hour baseline."""
	now = datetime.now(timezone.utc)
	window = timedelta(minutes=30)
	baseline_start = now - window - timedelta(hours=24)
	for bin_index in range(48):
		bin_start = baseline_start + bin_index * window
		for duplicate in range(2):
			_add_conversation(
				analytics_session,
				f"BASE-{bin_index}-{duplicate}",
				"order_status",
				bin_start + timedelta(minutes=5 + duplicate),
			)
	for index in range(7):
		_add_conversation(
			analytics_session,
			f"CURRENT-{index}",
			"order_status",
			now - timedelta(minutes=10, seconds=index),
		)
	analytics_session.commit()

	alerts = detect_root_cause_spikes(analytics_session, now=now)

	assert len(alerts) == 1
	assert alerts[0].count == 7
	assert alerts[0].baseline == 2.0


def test_root_cause_does_not_alert_at_absolute_threshold_or_on_stale_data(
	analytics_session: Session,
) -> None:
	"""Require strictly more than ten current reports and ignore older windows."""
	now = datetime.now(timezone.utc)
	for index in range(10):
		_add_conversation(
			analytics_session,
			f"CURRENT-{index}",
			"payment_failure",
			now - timedelta(minutes=15, seconds=index),
		)
	for index in range(12):
		_add_conversation(
			analytics_session,
			f"STALE-{index}",
			"fraud_complaint",
			now - timedelta(minutes=31, seconds=index),
		)
	analytics_session.commit()

	assert detect_root_cause_spikes(analytics_session, now=now) == []


def test_root_cause_keeps_intents_separate(
	analytics_session: Session,
) -> None:
	"""Do not combine independent six-conversation intents into one spike."""
	now = datetime.now(timezone.utc)
	for intent in ("payment_failure", "order_status"):
		for index in range(6):
			_add_conversation(
				analytics_session,
				f"{intent}-{index}",
				intent,
				now - timedelta(minutes=10, seconds=index),
			)
	analytics_session.commit()

	assert detect_root_cause_spikes(analytics_session, now=now) == []


def test_sla_check_flags_tier_windows_and_raises_priority(
	analytics_session: Session,
) -> None:
	"""Flag premium/standard tickets in their final 20 percent, not a distant ticket."""
	now = datetime.now(timezone.utc)
	premium = _add_ticket(analytics_session, "C001", sla_deadline=now + timedelta(minutes=2))
	standard = _add_ticket(analytics_session, "C002", sla_deadline=now + timedelta(minutes=11))
	far = _add_ticket(analytics_session, "C002", sla_deadline=now + timedelta(minutes=30))
	analytics_session.commit()

	flagged = sla_check(analytics_session, now=now)
	flagged_ids = {item["ticket_id"] for item in flagged}

	assert flagged_ids == {premium.id, standard.id}
	assert premium.sla_risk and standard.sla_risk
	assert premium.priority == standard.priority == "high"
	assert far.sla_risk is False
	assert sla_check(analytics_session, now=now) == []


def test_sla_check_flags_overdue_tickets(
	analytics_session: Session,
) -> None:
	"""Treat an overdue open ticket as SLA risk and flag it exactly once."""
	now = datetime.now(timezone.utc)
	ticket = _add_ticket(analytics_session, "C002", sla_deadline=now - timedelta(seconds=1))
	analytics_session.commit()

	flagged = sla_check(analytics_session, now=now)

	assert [item["ticket_id"] for item in flagged] == [ticket.id]
	assert ticket.sla_risk is True
	assert ticket.priority == "high"