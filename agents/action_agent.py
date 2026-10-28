"""Coordinate deterministic action tools and report only verified outcomes."""

import json
import logging
from collections.abc import Mapping
from typing import Any, Callable

from sqlalchemy.orm import Session

from agents.schemas import ActionMessage, ActionResult, ExecutedAction, ExtractedEntity
from llm.client import call_llm_json
from tools import action_tools

ACTION_MESSAGE_PROMPT = """Write a concise, empathetic customer message using only
the supplied action tool results. Do not claim an action succeeded unless its result
has success=true. Do not add policy, order, or account facts not present in those
results."""
logger = logging.getLogger(__name__)

ACTION_SEQUENCES: dict[str, tuple[str, ...]] = {
	"replace_and_return": (
		"create_return_request",
		"generate_return_label",
		"schedule_replacement",
		"send_notification",
	),
	"refund": ("initiate_refund", "send_notification"),
	"initiate_refund": ("initiate_refund", "send_notification"),
	"unlock_and_reset": ("unlock_account", "reset_password", "send_notification"),
	"cancel": ("cancel_order", "send_notification"),
	"cancel_order": ("cancel_order", "send_notification"),
	"create_return_request": ("create_return_request", "send_notification"),
	"generate_return_label": ("generate_return_label", "send_notification"),
	"schedule_replacement": ("schedule_replacement", "send_notification"),
	"unlock_account": ("unlock_account", "send_notification"),
	"reset_password": ("reset_password", "send_notification"),
}


def _entity_values(entities: Mapping[str, Any] | list[ExtractedEntity]) -> dict[str, str]:
	"""Normalize named entity records or a mapping into string values."""
	if isinstance(entities, Mapping):
		return {str(key): str(value) for key, value in entities.items()}
	return {entity.name: entity.value for entity in entities}


def _tool_call(
	tool_name: str,
	values: dict[str, str],
	customer_id: str,
	conversation_id: str,
	session: Session | None,
	message: str = "",
) -> dict[str, Any]:
	"""Call the requested action tool with normalized IDs and shared trace IDs."""
	tool: Callable[..., dict[str, Any]] = getattr(action_tools, tool_name)
	common = {"conversation_id": conversation_id, "session": session}
	if tool_name in {"create_return_request", "generate_return_label", "initiate_refund", "cancel_order", "schedule_replacement"}:
		order_id = values.get("order_id", "")
		if tool_name == "schedule_replacement":
			verified = values.get("verified", "").lower() in {"true", "yes", "verified"}
			return tool(order_id, verified=verified, **common)
		return tool(order_id, **common)
	if tool_name in {"unlock_account", "reset_password"}:
		return tool(customer_id, **common)
	if tool_name == "send_notification":
		return tool(customer_id, message, **common)
	return {"success": False, "data": {}, "error": f"Unsupported action tool: {tool_name}."}


def _executed_action(tool_name: str, result: dict[str, Any]) -> ExecutedAction:
	"""Convert a tool result into the public action trace schema."""
	if result["success"]:
		summary = json.dumps(result["data"], ensure_ascii=True, sort_keys=True)
	else:
		summary = str(result["error"] or "Action was blocked.")
	return ExecutedAction(tool_name=tool_name, success=result["success"], summary=summary)


def _template_message(results: list[dict[str, Any]]) -> str:
	"""Create customer wording from action results without adding new facts."""
	if not results:
		return "I could not perform that action because no supported tool result is available."
	failed = next((entry for entry in results if not entry["result"]["success"]), None)
	if failed is not None:
		error = failed["result"]["error"] or "the action was blocked by policy"
		return f"I could not complete {failed['tool_name']}: {error}"
	completed = ", ".join(entry["tool_name"].replace("_", " ") for entry in results)
	return f"Completed support actions: {completed}."


def _customer_message(results: list[dict[str, Any]]) -> str:
	"""Phrase a customer message from tool results, with a deterministic fallback."""
	if not results:
		return _template_message(results)
	try:
		response = call_llm_json(
			ACTION_MESSAGE_PROMPT,
			json.dumps(results, ensure_ascii=True, sort_keys=True),
			ActionMessage,
			"action",
		)
		return response.customer_message
	except Exception as error:
		logger.warning("Action message generation failed; using tool-result template: %s", error)
		return _template_message(results)


def execute(
	proposed_action: str,
	entities: Mapping[str, Any] | list[ExtractedEntity],
	customer_id: str,
	conversation_id: str,
	session: Session | None = None,
) -> ActionResult:
	"""Run a mapped action sequence, stopping at the first failed tool result."""
	sequence = ACTION_SEQUENCES.get(proposed_action)
	if sequence is None:
		return ActionResult(
			actions_executed=[],
			success=False,
			customer_message=f"No supported action is mapped for '{proposed_action}'.",
		)
	values = _entity_values(entities)
	tool_results: list[dict[str, Any]] = []
	executed: list[ExecutedAction] = []
	blocked = False
	for tool_name in sequence:
		if tool_name == "send_notification":
			continue
		result = _tool_call(
			tool_name,
			values,
			customer_id,
			conversation_id,
			session,
		)
		tool_results.append({"tool_name": tool_name, "result": result})
		executed.append(_executed_action(tool_name, result))
		if not result["success"]:
			blocked = True
			break

	customer_message = _customer_message(tool_results)
	if sequence[-1] == "send_notification" and not blocked:
		notification_result = _tool_call(
			"send_notification",
			values,
			customer_id,
			conversation_id,
			session,
			message=customer_message,
		)
		tool_results.append({"tool_name": "send_notification", "result": notification_result})
		executed.append(_executed_action("send_notification", notification_result))
		if not notification_result["success"]:
			blocked = True
			customer_message = _template_message(tool_results)

	return ActionResult(
		actions_executed=executed,
		success=bool(executed) and all(action.success for action in executed) and not blocked,
		customer_message=customer_message,
	)