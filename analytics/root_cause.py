"""Detect intent spikes in rolling conversation windows."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Conversation
from db.session import SessionLocal

DEFAULT_WINDOW_MINUTES = 30
DEFAULT_BASELINE_HOURS = 24
SPIKE_MULTIPLIER = 3.0
ABSOLUTE_SPIKE_THRESHOLD = 10


class RootCauseAlert(BaseModel):
	"""A high-volume intent cluster and its likely operational cause."""

	type: str = "root_cause_spike"
	intent: str
	count: int
	baseline: float
	window_minutes: int
	window_start: datetime
	window_end: datetime
	message: str


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a supplied database session or open a managed session."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _utc(value: datetime) -> datetime:
	"""Normalize database timestamps to UTC for window comparison."""
	if value.tzinfo is None:
		return value.replace(tzinfo=timezone.utc)
	return value.astimezone(timezone.utc)


def _conversation_intent(state: dict[str, Any]) -> str | None:
	"""Extract an intent from current or persisted memory state."""
	memory_state = state.get("memory_state", {})
	intent = memory_state.get("intent", state.get("intent"))
	if isinstance(intent, dict):
		intent = intent.get("intent")
	return str(intent) if intent else None


def detect_root_cause_spikes(
	session: Session | None = None,
	*,
	now: datetime | None = None,
	window_minutes: int = DEFAULT_WINDOW_MINUTES,
	baseline_hours: int = DEFAULT_BASELINE_HOURS,
) -> list[RootCauseAlert]:
	"""Compare the current rolling intent window with prior half-hour baselines."""
	if window_minutes <= 0 or baseline_hours <= 0:
		raise ValueError("Window and baseline durations must be positive.")
	window_end = _utc(now or datetime.now(timezone.utc))
	window = timedelta(minutes=window_minutes)
	window_start = window_end - window
	baseline_start = window_start - timedelta(hours=baseline_hours)
	window_count = max(1, baseline_hours * 60 // window_minutes)

	with _using_session(session) as db:
		conversations = db.scalars(
			select(Conversation).where(
				Conversation.started_at >= baseline_start,
				Conversation.started_at <= window_end,
			)
		).all()

		current_counts: dict[str, int] = defaultdict(int)
		baseline_bins: dict[str, list[int]] = defaultdict(
			lambda: [0] * window_count
		)
		for conversation in conversations:
			started_at = _utc(conversation.started_at)
			intent = _conversation_intent(conversation.state or {})
			if not intent:
				continue
			if window_start <= started_at <= window_end:
				current_counts[intent] += 1
			elif baseline_start <= started_at < window_start:
				index = int((started_at - baseline_start).total_seconds() // window.total_seconds())
				if 0 <= index < window_count:
					baseline_bins[intent][index] += 1

		alerts: list[RootCauseAlert] = []
		for intent, count in current_counts.items():
			baseline_values = baseline_bins[intent]
			baseline = sum(baseline_values) / window_count
			is_spike = count > ABSOLUTE_SPIKE_THRESHOLD or (
				baseline > 0 and count > SPIKE_MULTIPLIER * baseline
			)
			if not is_spike:
				continue
			cause = (
				", possible payment gateway outage"
				if intent == "payment_failure"
				else ", possible shared support issue"
			)
			alerts.append(
				RootCauseAlert(
					intent=intent,
					count=count,
					baseline=round(baseline, 2),
					window_minutes=window_minutes,
					window_start=window_start,
					window_end=window_end,
					message=(
						f"Spike in {intent}: {count} in {window_minutes} min{cause}."
					),
				)
			)
		return sorted(alerts, key=lambda alert: alert.count, reverse=True)