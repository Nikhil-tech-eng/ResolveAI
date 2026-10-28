"""Apply hard escalation rules before asking Gemini about borderline cases."""

import json
import logging
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from agents.schemas import EscalationDecision
from llm.client import call_llm_json

ESCALATION_PROMPT = """For a case that has not triggered a hard escalation rule, decide
whether to auto_resolve, escalate, or ask_more_info. Use only supplied conversation,
workflow, and investigation facts. Always provide a concise reason and confidence."""
logger = logging.getLogger(__name__)
HIGH_RISK_INTENTS = {"fraud_complaint", "legal_request", "data_breach"}
VOLATILE_FACT_KEYS = {"created_at", "updated_at", "placed_at", "delivered_at", "timestamp", "sla_deadline"}


def _mapping(value: Any) -> dict[str, Any]:
	"""Convert a Pydantic model or mapping to a plain dictionary."""
	if isinstance(value, BaseModel):
		return value.model_dump(mode="python")
	if isinstance(value, Mapping):
		return dict(value)
	return {}


def _decision(reason: str, confidence: float = 1.0) -> EscalationDecision:
	"""Return a deterministic escalation result with a non-empty reason."""
	return EscalationDecision(
		decision="escalate",
		reason=reason or "A hard escalation policy was triggered.",
		confidence=max(0.0, min(1.0, confidence)),
	)


def _failed_steps(state: dict[str, Any]) -> int:
	"""Count explicit or textually evident failed troubleshooting steps."""
	if isinstance(state.get("failed_steps"), int):
		return state["failed_steps"]
	troubleshoot = _mapping(state.get("troubleshoot"))
	steps = state.get("steps_run", troubleshoot.get("steps_run", []))
	failed_markers = ("failed", "error", "not found", "blocked", "could not")
	count = 0
	for step in steps if isinstance(steps, list) else []:
		step_data = _mapping(step)
		finding = str(step_data.get("finding", "")).lower()
		if step_data.get("success") is False or any(marker in finding for marker in failed_markers):
			count += 1
	return count


def _sla_risk(state: dict[str, Any], context: dict[str, Any]) -> str | None:
	"""Return an SLA escalation reason if deadline or tier limit is at risk."""
	if state.get("sla_at_risk") is True:
		return "The conversation is marked at risk of breaching its SLA."
	deadline_value = state.get("sla_deadline")
	if isinstance(deadline_value, str):
		try:
			deadline_value = datetime.fromisoformat(deadline_value)
		except ValueError:
			deadline_value = None
	if isinstance(deadline_value, datetime):
		if deadline_value.tzinfo is None:
			deadline_value = deadline_value.replace(tzinfo=timezone.utc)
		if deadline_value <= datetime.now(timezone.utc):
			return "The response SLA deadline has passed."

	tier = str(state.get("tier", context.get("tier", "standard"))).lower()
	limit_minutes = 15 if tier == "premium" else 60
	started_value = state.get("sla_started_at", state.get("started_at"))
	if isinstance(started_value, str):
		try:
			started_value = datetime.fromisoformat(started_value)
		except ValueError:
			started_value = None
	if isinstance(started_value, datetime):
		if started_value.tzinfo is None:
			started_value = started_value.replace(tzinfo=timezone.utc)
		if datetime.now(timezone.utc) - started_value >= timedelta(minutes=limit_minutes):
			return f"The {tier} customer response SLA of {limit_minutes} minutes is at risk."
	return None


def _stable_finding(finding: str) -> str:
	"""Remove volatile database timestamps from cached LLM investigation input."""
	try:
		parsed = json.loads(finding)
	except (json.JSONDecodeError, TypeError):
		return re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?", "<timestamp>", finding)
	if not isinstance(parsed, dict):
		return finding
	stable = {key: value for key, value in parsed.items() if key not in VOLATILE_FACT_KEYS}
	return json.dumps(stable, ensure_ascii=True, sort_keys=True)

def _verified_sku_mismatch(steps: list[Any]) -> bool:
	"""True if the delivered SKU is not among the ordered SKUs."""
	ordered: set[str] = set()
	delivered: set[str] = set()
	for step in steps:
		data = _mapping(step)
		try:
			finding = json.loads(str(data.get("finding", "")))
		except json.JSONDecodeError:
			continue
		if data.get("step") == "get_order_items":
			ordered = {item["sku"] for item in finding.get("items", []) if "sku" in item}
		elif data.get("step") == "get_shipment":
			delivered = {
				s["delivered_item_sku"]
				for s in finding.get("shipments", [])
				if s.get("delivered_item_sku")
			}
	return bool(ordered and delivered and not delivered <= ordered)

def decide(state: Mapping[str, Any] | BaseModel) -> EscalationDecision:
	"""Apply mandatory escalation rules first and call Gemini only if borderline."""
	state_data = _mapping(state)
	intent_data = _mapping(state_data.get("intent"))
	intent = str(intent_data.get("intent", state_data.get("intent", "general_query")))
	confidence_value = state_data.get("confidence")
	confidence = float(confidence_value) if confidence_value is not None else 1.0
	context = _mapping(state_data.get("context"))
	sentiment = str(
		intent_data.get("sentiment", state_data.get("sentiment", "neutral"))
	).lower()

	if intent in HIGH_RISK_INTENTS:
		return _decision(f"Intent '{intent}' requires a human specialist.", confidence)
	if confidence < 0.70:
		return _decision(f"Confidence {confidence:.2f} is below the 0.70 auto-resolution threshold.", confidence)

	resolved = bool(state_data.get("resolved", state_data.get("issue_resolved", False)))
	unresolved = bool(state_data.get("unresolved", not resolved))
	attempts = int(
		state_data.get(
			"attempts",
			state_data.get("attempt_count", state_data.get("troubleshooting_attempts", 0)),
		)
	)
	if unresolved and sentiment == "angry" and attempts >= 1:
		return _decision("The angry customer issue remains unresolved after one attempt.", confidence)
	if unresolved and sentiment == "negative" and attempts >= 2:
		return _decision("The strongly negative issue remains unresolved after two attempts.", confidence)
	if state_data.get("policy_blocked") and state_data.get(
		"customer_disputes", state_data.get("customer_disputed", False)
	):
		return _decision("The customer disputes an action blocked by policy.", confidence)
	sla_reason = _sla_risk(state_data, context)
	if sla_reason:
		return _decision(sla_reason, confidence)
	if _failed_steps(state_data) >= 3:
		return _decision("Three or more troubleshooting steps failed.", confidence)

	if intent == "wrong_item":
		troubleshoot_data = _mapping(state_data.get("troubleshoot"))
		steps = troubleshoot_data.get("steps_run", state_data.get("steps_run", []))
		if isinstance(steps, list) and _verified_sku_mismatch(steps):
			return EscalationDecision(
				decision="auto_resolve",
				reason="Delivered SKU does not match the ordered SKU; wrong item verified.",
				confidence=confidence,
			)
	if intent == "cancel_order" and state_data.get("policy_blocked"):
		return EscalationDecision(
			decision="auto_resolve",
			reason="Cancellation is blocked by policy; explain it to the customer.",
			confidence=confidence,
		)

	try:
		troubleshoot_data = _mapping(state_data.get("troubleshoot"))
		steps = troubleshoot_data.get("steps_run", state_data.get("steps_run", []))
		intent_data = _mapping(state_data.get("intent"))
		kb_data = _mapping(state_data.get("kb"))
		stable_facts = {
			"intent": intent,
			"category": intent_data.get("category", "general"),
			"priority": intent_data.get("priority", "medium"),
			"sentiment": sentiment,
			"confidence": confidence,
			"tier": state_data.get("tier", context.get("tier", "standard")),
			"workflow_id": kb_data.get("selected_workflow", "none"),
			"attempts": attempts,
			"unresolved": unresolved,
			"policy_blocked": bool(state_data.get("policy_blocked", False)),
			"customer_disputes": bool(
				state_data.get("customer_disputes", state_data.get("customer_disputed", False))
			),
			"steps_run": [
				{
					"step": _mapping(step).get("step", "check"),
					"finding": _stable_finding(str(_mapping(step).get("finding", ""))),
				}
				for step in steps
				if isinstance(steps, list)
			],
			"proposed_action": troubleshoot_data.get(
				"proposed_action", state_data.get("proposed_action", "")
			),
		}
		result = call_llm_json(
			ESCALATION_PROMPT,
			json.dumps(stable_facts, default=str, ensure_ascii=True),
			EscalationDecision,
			"escalation",
		)
		if not result.reason.strip():
			result.reason = "Borderline case decision based on the available investigation."
		return result
	except Exception as error:
		logger.warning("Borderline escalation decision failed; using safe deterministic default: %s", error)
		return EscalationDecision(
			decision="escalate",
			reason="The borderline case could not be evaluated reliably; human review is required.",
			confidence=confidence,
		)