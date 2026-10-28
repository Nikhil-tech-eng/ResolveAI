"""Generate complete human-ready tickets and persist them with SLA deadlines."""

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.schemas import (
	ExecutedAction,
	IntentName,
	Priority,
	Sentiment,
	Ticket,
	TicketText,
)
from db.models import ActionLog, Customer, Ticket as DatabaseTicket
from db.session import SessionLocal
from llm.client import call_llm_json

TICKET_TEXT_PROMPT = """Write a factual one- or two-sentence issue summary and one
useful suggested next step for a human support agent. Use only the supplied customer
statement, verified troubleshooting findings, and action-log results. Do not claim
an action was taken unless the action log says it succeeded."""
logger = logging.getLogger(__name__)


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a supplied DB session or create a short-lived managed session."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _as_dict(value: Any) -> dict[str, Any]:
	"""Convert a Pydantic value or mapping to a standard dictionary."""
	if hasattr(value, "model_dump"):
		return value.model_dump(mode="python")
	if isinstance(value, Mapping):
		return dict(value)
	return {}


def _text_from_messages(messages: Any) -> tuple[str, str]:
	"""Extract the latest customer message and a short conversation summary."""
	if not isinstance(messages, list):
		return "", "No conversation messages were supplied."
	message_texts: list[str] = []
	customer_text: list[str] = []
	for message in messages:
		if isinstance(message, str):
			message_texts.append(message)
			customer_text.append(message)
			continue
		message_data = _as_dict(message)
		content = str(message_data.get("content", ""))
		if content:
			message_texts.append(content)
			if message_data.get("role") == "customer":
				customer_text.append(content)
	return (customer_text[-1] if customer_text else "", " ".join(message_texts[-5:]))


def _intent_fields(state: dict[str, Any]) -> tuple[IntentName, str, Priority, Sentiment]:
	"""Normalize intent, category, priority, and sentiment from graph state."""
	intent_data = _as_dict(state.get("intent"))
	intent_value = str(intent_data.get("intent", state.get("intent", "general_query")))
	valid_intents = {
		"wrong_item", "damaged_item", "missing_item", "order_status", "payment_failure",
		"refund_request", "cancel_order", "login_issue", "subscription_issue",
		"fraud_complaint", "legal_request", "data_breach", "general_query",
	}
	intent: IntentName = intent_value if intent_value in valid_intents else "general_query"  # type: ignore[assignment]
	category = str(intent_data.get("category", state.get("category", "general")))
	priority_value = str(intent_data.get("priority", state.get("priority", "medium")))
	priority: Priority = priority_value if priority_value in {"low", "medium", "high"} else "medium"  # type: ignore[assignment]
	sentiment_value = str(intent_data.get("sentiment", state.get("sentiment", "neutral")))
	sentiment: Sentiment = sentiment_value if sentiment_value in {"positive", "neutral", "negative", "angry"} else "neutral"  # type: ignore[assignment]
	return intent, category, priority, sentiment


def _entity_value(entities: Any, name: str) -> str | None:
	"""Find a named entity from a map or list of entity records."""
	if isinstance(entities, Mapping):
		value = entities.get(name)
		return str(value) if value is not None else None
	if isinstance(entities, list):
		for entity in entities:
			entity_data = _as_dict(entity)
			if entity_data.get("name") == name:
				return str(entity_data.get("value", ""))
	return None


def _fallback_ticket_text(
	intent: str,
	customer_message: str,
	steps: list[dict[str, Any]],
	actions: list[dict[str, Any]],
	escalation_reason: str,
) -> TicketText:
	"""Create factual summary and next step if ticket phrasing fails."""
	issue_summary = customer_message.strip() or f"Customer needs assistance with {intent.replace('_', ' ')}."
	if len(issue_summary) > 300:
		issue_summary = issue_summary[:297].rstrip() + "..."
	if escalation_reason:
		next_step = f"Review escalation: {escalation_reason}"
	elif any(action.get("success") is False for action in actions):
		next_step = "Review the blocked action and determine an allowed resolution."
	elif steps:
		next_step = "Review the recorded investigation findings and continue the verified workflow."
	else:
		next_step = "Review the conversation and confirm the next support action."
	return TicketText(issue_summary=issue_summary, suggested_next_step=next_step)


def generate_ticket(
	state: Mapping[str, Any] | Any,
	session: Session | None = None,
) -> Ticket:
	"""Create a human-ready ticket from graph state and action logs, then persist it."""
	state_data = _as_dict(state)
	customer_id = str(state_data.get("customer_id", ""))
	conversation_id = str(state_data.get("conversation_id", ""))
	if not customer_id:
		raise ValueError("Ticket state must include customer_id.")
	if not conversation_id:
		raise ValueError("Ticket state must include conversation_id.")

	intent, category, priority, sentiment = _intent_fields(state_data)
	context_data = _as_dict(state_data.get("context"))
	tier = str(context_data.get("tier", "standard"))
	profile_data = _as_dict(context_data.get("profile"))
	customer_message, conversation_summary = _text_from_messages(state_data.get("messages", []))
	troubleshoot_data = _as_dict(state_data.get("troubleshoot"))
	steps_value = troubleshoot_data.get("steps_run", state_data.get("steps_run", []))
	steps = [_as_dict(step) for step in steps_value] if isinstance(steps_value, list) else []
	escalation_data = _as_dict(state_data.get("escalation"))
	escalation_reason = str(escalation_data.get("reason", state_data.get("escalation_reason", "")))

	with _using_session(session) as db:
		customer = db.get(Customer, customer_id)
		if customer is None:
			raise ValueError(f"Customer {customer_id} was not found.")
		tier = customer.tier
		logs = db.scalars(
			select(ActionLog)
			.where(ActionLog.conversation_id == conversation_id)
			.order_by(ActionLog.timestamp, ActionLog.id)
		).all()
		actions = [
			{
				"tool_name": log.tool_name,
				"success": log.success,
				"summary": json.dumps(log.result, ensure_ascii=True, sort_keys=True),
			}
			for log in logs
		]
		order_id = _entity_value(state_data.get("entities"), "order_id")
		if order_id is None:
			order_id = _entity_value(_as_dict(state_data.get("intent")).get("extracted_entities"), "order_id")
		created_at = datetime.now(timezone.utc)
		sla_minutes = 15 if tier == "premium" else 60
		sla_deadline = created_at + timedelta(minutes=sla_minutes)
		facts = {
			"customer_id": customer_id,
			"customer_name": customer.name,
			"customer_tier": tier,
			"order_id": order_id,
			"intent": intent,
			"category": category,
			"priority": priority,
			"sentiment": sentiment,
			"customer_statement": customer_message,
			"investigation_steps": steps,
			"action_log_results": actions,
			"escalation_reason": escalation_reason,
		}
		try:
			text_result = call_llm_json(
				TICKET_TEXT_PROMPT,
				json.dumps(facts, ensure_ascii=True),
				TicketText,
				"ticket",
			)
		except Exception as error:
			logger.warning("Ticket text generation failed; using factual template: %s", error)
			text_result = _fallback_ticket_text(
				intent, customer_message, steps, actions, escalation_reason
			)

		investigation = "\n".join(
			f"{step.get('step', 'check')}: {step.get('finding', '')}" for step in steps
		) or "No troubleshooting steps were recorded."
		assigned_to = (
			"support_queue" if escalation_data.get("decision") == "escalate" else None
		)
		row = DatabaseTicket(
			customer_id=customer_id,
			order_id=order_id,
			intent=intent,
			category=category,
			priority=priority,
			sentiment=sentiment,
			summary=text_result.issue_summary,
			investigation=investigation,
			actions_taken=actions,
			status="opened",
			assigned_to=assigned_to,
			escalation_reason=escalation_reason or None,
			sla_deadline=sla_deadline,
			created_at=created_at,
			updated_at=created_at,
		)
		db.add(row)
		db.flush()
		ticket = Ticket(
			ticket_id=row.id,
			customer_id=customer_id,
			customer_name=customer.name,
			customer_tier=tier,
			order_id=order_id,
			intent=intent,
			category=category,
			priority=priority,
			sentiment=sentiment,
			summary=text_result.issue_summary,
			investigation=investigation,
			actions_taken=[
				ExecutedAction(**action) for action in actions
			],
			status="opened",
			assigned_to=assigned_to,
			escalation_reason=escalation_reason or None,
			conversation_summary=conversation_summary,
			suggested_next_step=text_result.suggested_next_step,
			sla_deadline=sla_deadline.isoformat(),
		)
		db.commit()
		return ticket