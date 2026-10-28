"""Validate ticket lifecycle transitions and monitor SLA risk."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from db.models import Ticket
from db.session import SessionLocal
from tools.action_tools import send_notification

logger = logging.getLogger(__name__)

OPEN_STATUSES = ("opened", "assigned", "pending_customer")
SLA_MINUTES_BY_TIER = {"premium": 15, "standard": 60}
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
	"opened": {"assigned"},
	"assigned": {"pending_customer"},
	"pending_customer": {"assigned", "resolved"},
	"resolved": {"closed", "opened"},
	"closed": set(),
}
STATUS_MESSAGES = {
	"opened": "Your support ticket has been reopened.",
	"assigned": "Your support ticket has been assigned to our support team.",
	"pending_customer": "We need information from you to continue your support ticket.",
	"resolved": "Your support ticket has been marked resolved.",
	"closed": "Your support ticket has been closed.",
}


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a caller-provided database session or open a managed one."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _utc(value: datetime) -> datetime:
	"""Normalize a datetime to timezone-aware UTC."""
	if value.tzinfo is None:
		return value.replace(tzinfo=timezone.utc)
	return value.astimezone(timezone.utc)


def _ensure_sla_risk_column(session: Session) -> None:
	"""Add the SLA flag for existing SQLite databases created before this field."""
	bind = session.get_bind()
	if bind.dialect.name != "sqlite":
		return
	column_names = {column["name"] for column in inspect(bind).get_columns("tickets")}
	if "sla_risk" not in column_names:
		session.execute(
			text("ALTER TABLE tickets ADD COLUMN sla_risk BOOLEAN NOT NULL DEFAULT 0")
		)
		session.commit()


def transition(
	ticket_id: int,
	new_status: str,
	actor: str,
	session: Session | None = None,
) -> dict[str, Any]:
	"""Validate and persist one ticket status transition, then notify its customer."""
	with _using_session(session) as db:
		_ensure_sla_risk_column(db)
		ticket = db.get(Ticket, ticket_id)
		if ticket is None:
			raise ValueError(f"Ticket {ticket_id} was not found.")
		if new_status not in ALLOWED_TRANSITIONS.get(ticket.status, set()):
			allowed = ", ".join(sorted(ALLOWED_TRANSITIONS.get(ticket.status, set()))) or "none"
			raise ValueError(
				f"Invalid ticket transition: {ticket.status} -> {new_status}. "
				f"Allowed next status: {allowed}."
			)
		old_status = ticket.status
		ticket.status = new_status
		db.commit()
		message = (
			f"{STATUS_MESSAGES[new_status]} Ticket #{ticket.id}. "
			f"Status updated by {actor}."
		)
		notification = send_notification(
			ticket.customer_id,
			message,
			session=db,
		)
		if not notification["success"]:
			logger.warning("Ticket %s changed status but notification failed: %s", ticket.id, notification["error"])
		return {
			"ticket_id": ticket.id,
			"old_status": old_status,
			"new_status": ticket.status,
			"actor": actor,
			"notification": notification,
		}


def sla_check(
	session: Session | None = None,
	now: datetime | None = None,
) -> list[dict[str, Any]]:
	"""Flag open tickets within the final 20 percent of their tier SLA."""
	current_time = _utc(now or datetime.now(timezone.utc))
	flagged: list[dict[str, Any]] = []
	with _using_session(session) as db:
		_ensure_sla_risk_column(db)
		tickets = db.scalars(
			select(Ticket)
			.where(Ticket.status.in_(OPEN_STATUSES), Ticket.sla_deadline.is_not(None))
			.order_by(Ticket.sla_deadline, Ticket.id)
		).all()
		for ticket in tickets:
			deadline = _utc(ticket.sla_deadline)
			customer_tier = ticket.customer.tier.lower()
			sla_duration = timedelta(
				minutes=SLA_MINUTES_BY_TIER.get(customer_tier, SLA_MINUTES_BY_TIER["standard"])
			)
			risk_window = sla_duration * 0.20
			if deadline - current_time <= risk_window and not ticket.sla_risk:
				ticket.sla_risk = True
				ticket.priority = "high"
				flagged.append(
					{
						"ticket_id": ticket.id,
						"customer_id": ticket.customer_id,
						"sla_deadline": deadline.isoformat(),
						"priority": ticket.priority,
						"sla_risk": ticket.sla_risk,
					}
				)
		db.commit()
	return flagged