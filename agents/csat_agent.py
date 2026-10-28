"""Collect and aggregate customer satisfaction feedback."""

import logging
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agents.schemas import CSATResult, Sentiment
from db.models import CSAT, ActionLog, Conversation, Ticket
from db.session import SessionLocal
from llm.client import call_llm_json

COMMENT_SENTIMENT_PROMPT = """Classify only the sentiment expressed in the customer feedback
comment as positive, neutral, negative, or angry. Do not infer issue resolution quality."""
logger = logging.getLogger(__name__)
POSITIVE_WORDS = {
	"good", "great", "helpful", "thanks", "thank", "happy", "satisfied", "resolved", "excellent", "awesome"
}
NEGATIVE_WORDS = {
	"bad", "poor", "unhappy", "disappointed", "upset", "frustrated", "unresolved", "terrible", "waste"
}
ANGRY_WORDS = {"angry", "furious", "ridiculous", "unacceptable", "outraged", "gussa"}


class CommentSentiment(BaseModel):
	"""Validated sentiment classification for one CSAT comment."""

	sentiment: Sentiment


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a caller-provided database session or open a managed one."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _fallback_sentiment(comment: str) -> Sentiment:
	"""Classify comment sentiment using deterministic keyword sets."""
	words = set(re.findall(r"[a-z]+", comment.lower()))
	if words & ANGRY_WORDS:
		return "angry"
	if words & NEGATIVE_WORDS:
		return "negative"
	if words & POSITIVE_WORDS:
		return "positive"
	return "neutral"


def _comment_sentiment(comment: str) -> Sentiment:
	"""Ask Gemini to classify non-empty feedback, with a keyword fallback."""
	if not comment.strip():
		return "neutral"
	try:
		result = call_llm_json(
			COMMENT_SENTIMENT_PROMPT, comment, CommentSentiment, "csat"
		)
		return result.sentiment
	except Exception as error:
		logger.warning("CSAT sentiment call failed; using keyword classifier: %s", error)
		return _fallback_sentiment(comment)


def _resolution_quality(
	conversation: Conversation,
	ticket: Ticket | None,
	tool_action_logs: list[ActionLog],
) -> Literal["resolved_fully", "partially", "not_resolved"]:
	"""Determine resolution quality from persisted conversation and action state."""
	state = conversation.state or {}
	memory = state.get("memory_state", {})
	lifecycle_stage = memory.get("lifecycle_stage", state.get("lifecycle_stage"))
	if (
		state.get("resolved") is True
		or lifecycle_stage == "resolved"
		or conversation.status in {"resolved", "closed"}
		or (ticket is not None and ticket.status in {"resolved", "closed"})
	):
		return "resolved_fully"
	if lifecycle_stage in {"acting", "investigating"} or any(
		log.success for log in tool_action_logs
	):
		return "partially"
	return "not_resolved"


def collect(
	conversation_id: str,
	rating: int,
	comment: str | None,
	session: Session | None = None,
) -> CSATResult:
	"""Classify and persist feedback for an existing conversation."""
	with _using_session(session) as db:
		conversation = db.get(Conversation, conversation_id)
		if conversation is None:
			raise ValueError(f"Conversation {conversation_id} was not found.")
		state = conversation.state or {}
		ticket_id = state.get("ticket_id")
		ticket = db.get(Ticket, ticket_id) if ticket_id is not None else None
		if ticket is None:
			ticket = db.scalar(
				select(Ticket)
				.where(Ticket.customer_id == conversation.customer_id)
				.order_by(Ticket.created_at.desc(), Ticket.id.desc())
			)
		action_logs = db.scalars(
			select(ActionLog).where(ActionLog.conversation_id == conversation_id)
		).all()
		quality = _resolution_quality(conversation, ticket, action_logs)
		result = CSATResult(
			rating=rating,
			sentiment="neutral",
			resolution_quality=quality,
		)
		result.sentiment = _comment_sentiment(comment or "")
		db.add(
			CSAT(
				ticket_id=ticket.id if ticket is not None else None,
				conversation_id=conversation_id,
				rating=result.rating,
				sentiment=result.sentiment,
				resolution_quality=result.resolution_quality,
				comment=comment,
			)
		)
		db.commit()
		return result


def csat_summary(session: Session | None = None) -> dict[str, Any]:
	"""Return rating average and sentiment/resolution-quality aggregates."""
	with _using_session(session) as db:
		count, average = db.execute(select(func.count(CSAT.id), func.avg(CSAT.rating))).one()
		sentiments = db.execute(
			select(CSAT.sentiment, func.count(CSAT.id)).group_by(CSAT.sentiment)
		).all()
		qualities = db.execute(
			select(CSAT.resolution_quality, func.count(CSAT.id)).group_by(
				CSAT.resolution_quality
			)
		).all()
		return {
			"count": int(count),
			"average_rating": round(float(average), 2) if average is not None else None,
			"sentiment_distribution": dict(Counter({str(key): int(value) for key, value in sentiments})),
			"resolution_quality_distribution": {
				str(key): int(value) for key, value in qualities
			},
		}