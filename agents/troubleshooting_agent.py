"""Run deterministic support checks and use Gemini only to phrase diagnosis."""

import json
import logging
import re
from collections.abc import Callable, Mapping

from sqlalchemy.orm import Session

from agents.schemas import (
	DiagnosisResponse,
	ExtractedEntity,
	IntentResult,
	TroubleshootResult,
	TroubleshootStep,
)
from llm.client import call_llm_json
from tools import read_tools

DIAGNOSIS_PROMPT = """Write one concise, empathetic diagnosis based only on the
provided deterministic troubleshooting findings. Do not claim an action has already
been performed. Do not invent order, payment, account, or delivery details."""
logger = logging.getLogger(__name__)
ORDER_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)


def _entity_values(entities: Mapping[str, str] | list[ExtractedEntity]) -> dict[str, str]:
	"""Normalize entity mappings or schema entities into string values."""
	if isinstance(entities, Mapping):
		return {str(key): str(value) for key, value in entities.items()}
	return {entity.name: entity.value for entity in entities}


def _order_id(values: dict[str, str]) -> str | None:
	"""Find an explicit order ID in extracted entities."""
	for key, value in values.items():
		if key.lower() in {"order_id", "orderid"}:
			match = ORDER_PATTERN.search(value)
			if match:
				return match.group(0).upper()
	return None


def _read_step(
	tool_name: str,
	tool: Callable[..., dict[str, object]],
	identifier: str,
	session: Session | None,
) -> tuple[TroubleshootStep, dict[str, object]]:
	"""Call a read tool and capture its structured facts in a troubleshooting step."""
	result = tool(identifier, session=session)
	if not result["success"]:
		finding = str(result["error"] or f"{tool_name} lookup failed.")
	else:
		finding = json.dumps(result["data"], ensure_ascii=True, sort_keys=True)
	return TroubleshootStep(step=tool_name, finding=finding), result


def _run_wrong_item(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Verify ordered and delivered SKUs before proposing a replacement."""
	order_id = _order_id(values)
	if order_id is None:
		return [TroubleshootStep(step="identify_order", finding="Order ID is missing.")], "ask_customer_for_order_id"
	steps: list[TroubleshootStep] = []
	order_step, order = _read_step("get_order", read_tools.get_order, order_id, session)
	steps.append(order_step)
	if not order["success"]:
		return steps, "ask_customer_to_confirm_order"
	items_step, items = _read_step("get_order_items", read_tools.get_order_items, order_id, session)
	shipment_step, shipment = _read_step("get_shipment", read_tools.get_shipment, order_id, session)
	steps.extend((items_step, shipment_step))
	ordered_skus = {item["sku"] for item in items["data"].get("items", [])} if items["success"] else set()
	shipments = shipment["data"].get("shipments", []) if shipment["success"] else []
	delivered_skus = {item.get("delivered_item_sku") for item in shipments}
	if ordered_skus and delivered_skus and not (ordered_skus & delivered_skus):
		return steps, "schedule_replacement"
	return steps, "request_item_photo_and_verify"


def _run_damaged_item(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Check order existence and require explicit evidence verification."""
	order_id = _order_id(values)
	if order_id is None:
		return [TroubleshootStep(step="identify_order", finding="Order ID is missing.")], "ask_customer_for_order_id"
	steps, order = _read_step("get_order", read_tools.get_order, order_id, session)
	if not order["success"]:
		return [steps], "ask_customer_to_confirm_order"
	verified = values.get("verified", "").lower() in {"true", "yes", "verified"}
	steps.append(
		TroubleshootStep(
			step="verify_damage",
			finding="Customer evidence is marked verified." if verified else "Damage has not been verified from evidence.",
		)
	)
	return steps, "schedule_replacement" if verified else "request_damage_photos"


def _run_payment_deducted(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Inspect payment logs for a failed transaction that deducted funds."""
	order_id = _order_id(values)
	if order_id is None:
		return [TroubleshootStep(step="identify_order", finding="Order ID is missing.")], "ask_customer_for_order_id"
	order_step, order = _read_step("get_order", read_tools.get_order, order_id, session)
	payment_step, payment_result = _read_step("get_payment_logs", read_tools.get_payment_logs, order_id, session)
	steps = [order_step, payment_step]
	payments = payment_result["data"].get("payments", []) if payment_result["success"] else []
	deducted = any(payment.get("amount_deducted") for payment in payments)
	return steps, "initiate_refund" if deducted else "verify_payment_with_gateway"


def _run_account_locked(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Read customer account state before proposing an unlock."""
	customer_id = values.get("customer_id", "")
	if not customer_id:
		return [TroubleshootStep(step="identify_customer", finding="Customer ID is missing.")], "ask_customer_for_account_identifier"
	step, customer = _read_step("get_customer", read_tools.get_customer, customer_id, session)
	if customer["success"] and customer["data"].get("account_status") == "locked":
		return [step], "unlock_account"
	return [step], "verify_account_status"


def _run_order_status(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Look up the current order and tracking status."""
	order_id = _order_id(values)
	if order_id is None:
		return [TroubleshootStep(step="identify_order", finding="Order ID is missing.")], "ask_customer_for_order_id"
	order_step, order = _read_step("get_order", read_tools.get_order, order_id, session)
	shipment_step, _ = _read_step("get_shipment", read_tools.get_shipment, order_id, session)
	return [order_step, shipment_step], "share_verified_order_status" if order["success"] else "ask_customer_to_confirm_order"


def _run_cancel_order(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Check an order's state against the placed-only cancellation policy."""
	order_id = _order_id(values)
	if order_id is None:
		return [TroubleshootStep(step="identify_order", finding="Order ID is missing.")], "ask_customer_for_order_id"
	step, order = _read_step("get_order", read_tools.get_order, order_id, session)
	if not order["success"]:
		return [step], "ask_customer_to_confirm_order"
	if order["data"].get("status") == "placed":
		return [step], "cancel_order"
	return [step], "escalate_policy_block"


def _run_subscription_sync(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Read the subscription status and detect a paid-but-inactive sync state."""
	customer_id = values.get("customer_id", "")
	if not customer_id:
		return [TroubleshootStep(step="identify_customer", finding="Customer ID is missing.")], "ask_customer_for_account_identifier"
	step, subscription = _read_step("get_subscription", read_tools.get_subscription, customer_id, session)
	status = subscription["data"].get("subscription_status") if subscription["success"] else None
	action = "escalate_subscription_sync" if status == "paid_not_activated_sync_failed" else "verify_subscription_status"
	return [step], action


def _run_refund_request(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Verify payment and delivery facts before proposing refund or escalation."""
	order_id = _order_id(values)
	if order_id is None:
		return [TroubleshootStep(step="identify_order", finding="Order ID is missing.")], "ask_customer_for_order_id"
	order_step, order = _read_step("get_order", read_tools.get_order, order_id, session)
	payment_step, payments_result = _read_step("get_payment_logs", read_tools.get_payment_logs, order_id, session)
	steps = [order_step, payment_step]
	if not order["success"]:
		return steps, "ask_customer_to_confirm_order"
	payments = payments_result["data"].get("payments", []) if payments_result["success"] else []
	deducted = any(payment.get("amount_deducted") for payment in payments)
	if order["data"].get("payment_status") == "paid" or deducted:
		return steps, "initiate_refund"
	return steps, "escalate_refund_policy"


def _run_password_reset(values: dict[str, str], session: Session | None) -> tuple[list[TroubleshootStep], str]:
	"""Check account state before proposing a reset or account unlock."""
	customer_id = values.get("customer_id", "")
	if not customer_id:
		return [TroubleshootStep(step="identify_customer", finding="Customer ID is missing.")], "ask_customer_for_account_identifier"
	step, customer = _read_step("get_customer", read_tools.get_customer, customer_id, session)
	if not customer["success"]:
		return [step], "ask_customer_to_confirm_account"
	status = customer["data"].get("account_status")
	if status == "locked":
		return [step], "unlock_account"
	if status == "active":
		return [step], "reset_password"
	return [step], "escalate_suspended_account"


WORKFLOW_RUNNERS: dict[
	str, Callable[[dict[str, str], Session | None], tuple[list[TroubleshootStep], str]]
] = {
	"wrong_item": _run_wrong_item,
	"damaged_item": _run_damaged_item,
	"payment_deducted": _run_payment_deducted,
	"account_locked": _run_account_locked,
	"order_status": _run_order_status,
	"cancel_order": _run_cancel_order,
	"subscription_sync": _run_subscription_sync,
	"refund_request": _run_refund_request,
	"password_reset": _run_password_reset,
}


def _diagnosis_fallback(steps: list[TroubleshootStep], proposed_action: str) -> str:
	"""Phrase a concise deterministic diagnosis from recorded findings."""
	if proposed_action.startswith("ask_customer") or proposed_action.startswith("request_"):
		return "I need one more detail or piece of evidence to verify the issue before taking action."
	if proposed_action.startswith("escalate"):
		return "The records show this needs review by a support specialist before we proceed."
	if proposed_action in {"schedule_replacement", "initiate_refund", "cancel_order", "unlock_account", "reset_password"}:
		return f"I verified the relevant records. The next available step is {proposed_action.replace('_', ' ')}."
	if steps:
		return "I checked the available account and order records. " + steps[0].finding
	return "I could not verify enough information to diagnose this issue."


def troubleshoot(
	intent: str | IntentResult,
	entities: Mapping[str, str] | list[ExtractedEntity],
	workflow_id: str,
	session: Session | None = None,
) -> TroubleshootResult:
	"""Run the selected deterministic workflow and phrase its verified diagnosis."""
	values = _entity_values(entities)
	intent_name = intent.intent if isinstance(intent, IntentResult) else intent
	runner = WORKFLOW_RUNNERS.get(workflow_id)
	if runner is None:
		steps = [TroubleshootStep(step="select_workflow", finding=f"No runner exists for workflow '{workflow_id}'.")]
		proposed_action = "ask_more_info"
	else:
		steps, proposed_action = runner(values, session)
	facts = {
		"intent": intent_name,
		"workflow_id": workflow_id,
		"steps_run": [step.model_dump(mode="json") for step in steps],
		"proposed_action": proposed_action,
	}
	try:
		response = call_llm_json(
			DIAGNOSIS_PROMPT,
			json.dumps(facts, ensure_ascii=True),
			DiagnosisResponse,
			"troubleshooting",
		)
		diagnosis = response.diagnosis
	except Exception as error:
		logger.warning("Diagnosis phrasing failed; using deterministic template: %s", error)
		diagnosis = _diagnosis_fallback(steps, proposed_action)
	return TroubleshootResult(
		steps_run=steps,
		diagnosis=diagnosis,
		proposed_action=proposed_action,
	)