"""Deterministic support actions that enforce policy and persist action logs."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import ActionLog, Conversation, Customer, Message, Order, OrderItem, Payment, Shipment
from db.session import SessionLocal
from tools.policy import (
	check_cancel_policy,
	check_notification_policy,
	check_password_reset_policy,
	check_refund_policy,
	check_replacement_policy,
	check_return_policy,
	check_unlock_account_policy,
	return_shipping_fee,
)


class ToolResult(TypedDict):
	"""Common result shape returned by support tools."""

	success: bool
	data: dict[str, Any]
	error: str | None


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a caller session or open a short-lived session for one action."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _result(success: bool, data: dict[str, Any], error: str | None) -> ToolResult:
	"""Build a tool result with a consistent three-field shape."""
	return {"success": success, "data": data, "error": error}


def _finish_action(
	db: Session,
	*,
	tool_name: str,
	conversation_id: str | None,
	args: dict[str, Any],
	success: bool,
	data: dict[str, Any],
	error: str | None,
) -> ToolResult:
	"""Persist an action outcome and return its structured result."""
	db.add(
		ActionLog(
			conversation_id=conversation_id,
			tool_name=tool_name,
			args=args,
			result={"success": success, "data": data, "error": error},
			success=success,
		)
	)
	db.commit()
	return _result(success, data, error)


def create_return_request(
	order_id: str,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Create a policy-approved return request and log the outcome."""
	args = {"order_id": order_id}
	with _using_session(session) as db:
		order = db.get(Order, order_id)
		allowed, error = check_return_policy(order)
		if not allowed:
			return _finish_action(
				db,
				tool_name="create_return_request",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		customer = db.get(Customer, order.customer_id)
		data = {
			"return_request_id": f"RET-{order.id}",
			"order_id": order.id,
			"status": "requested",
			"return_shipping_fee": return_shipping_fee(customer),
		}
		return _finish_action(
			db,
			tool_name="create_return_request",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def generate_return_label(
	order_id: str,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Generate a return label only when the return policy permits it."""
	args = {"order_id": order_id}
	with _using_session(session) as db:
		order = db.get(Order, order_id)
		allowed, error = check_return_policy(order)
		if not allowed:
			return _finish_action(
				db,
				tool_name="generate_return_label",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		customer = db.get(Customer, order.customer_id)
		data = {
			"order_id": order.id,
			"label_id": f"LBL-{order.id}",
			"carrier": "Prepaid Returns",
			"shipping_fee": return_shipping_fee(customer),
		}
		return _finish_action(
			db,
			tool_name="generate_return_label",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def initiate_refund(
	order_id: str,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Refund an eligible order and log the payment action."""
	args = {"order_id": order_id}
	with _using_session(session) as db:
		order = db.get(Order, order_id)
		payments = (
			db.scalars(select(Payment).where(Payment.order_id == order_id)).all()
			if order is not None
			else []
		)
		allowed, error = check_refund_policy(order, payments)
		if not allowed:
			return _finish_action(
				db,
				tool_name="initiate_refund",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		order.payment_status = "refunded"
		for payment in payments:
			if payment.status == "paid" or payment.amount_deducted:
				payment.status = "refunded"
		data = {"order_id": order.id, "refund_amount": order.total, "status": "initiated"}
		return _finish_action(
			db,
			tool_name="initiate_refund",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def cancel_order(
	order_id: str,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Cancel an order only while its status is placed."""
	args = {"order_id": order_id}
	with _using_session(session) as db:
		order = db.get(Order, order_id)
		allowed, error = check_cancel_policy(order)
		if not allowed:
			return _finish_action(
				db,
				tool_name="cancel_order",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		order.status = "cancelled"
		data = {"order_id": order.id, "status": order.status}
		return _finish_action(
			db,
			tool_name="cancel_order",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def reset_password(
	customer_id: str,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Start a password reset for an eligible customer account."""
	args = {"customer_id": customer_id}
	with _using_session(session) as db:
		customer = db.get(Customer, customer_id)
		allowed, error = check_password_reset_policy(customer)
		if not allowed:
			return _finish_action(
				db,
				tool_name="reset_password",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		data = {"customer_id": customer.id, "reset_link_sent": True}
		return _finish_action(
			db,
			tool_name="reset_password",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def unlock_account(
	customer_id: str,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Unlock a locked customer account and reset its failed-attempt count."""
	args = {"customer_id": customer_id}
	with _using_session(session) as db:
		customer = db.get(Customer, customer_id)
		allowed, error = check_unlock_account_policy(customer)
		if not allowed:
			return _finish_action(
				db,
				tool_name="unlock_account",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		customer.account_status = "active"
		customer.failed_login_attempts = 0
		data = {"customer_id": customer.id, "account_status": customer.account_status}
		return _finish_action(
			db,
			tool_name="unlock_account",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def schedule_replacement(
	order_id: str,
	verified: bool = False,
	conversation_id: str | None = None,
	session: Session | None = None,
) -> ToolResult:
	"""Schedule a replacement only for a verified wrong or damaged item."""
	args = {"order_id": order_id, "verified": verified}
	with _using_session(session) as db:
		order = db.get(Order, order_id)
		items = (
			db.scalars(select(OrderItem).where(OrderItem.order_id == order_id)).all()
			if order is not None
			else []
		)
		shipment = db.scalar(
			select(Shipment)
			.where(Shipment.order_id == order_id)
			.order_by(Shipment.id.desc())
		)
		ordered_skus = {item.sku for item in items}
		wrong_item_verified = (
			shipment is not None
			and shipment.delivered_item_sku is not None
			and bool(ordered_skus)
			and shipment.delivered_item_sku not in ordered_skus
		)
		is_verified = verified or wrong_item_verified
		allowed, error = check_replacement_policy(is_verified)
		if order is None:
			allowed, error = False, "Order was not found."
		if not allowed:
			return _finish_action(
				db,
				tool_name="schedule_replacement",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		data = {
			"order_id": order.id,
			"replacement_id": f"RPL-{order.id}",
			"status": "scheduled",
			"verification": "wrong_item" if wrong_item_verified else "customer_verified",
		}
		return _finish_action(
			db,
			tool_name="schedule_replacement",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)


def send_notification(
	customer_id: str,
	message: str,
	conversation_id: str | None = None,
	channel: str = "email",
	session: Session | None = None,
) -> ToolResult:
	"""Record a notification for an existing customer and optional conversation."""
	args = {"customer_id": customer_id, "message": message, "channel": channel}
	with _using_session(session) as db:
		customer = db.get(Customer, customer_id)
		conversation = (
			db.get(Conversation, conversation_id) if conversation_id is not None else None
		)
		allowed, error = check_notification_policy(customer, conversation)
		if conversation_id is not None and conversation is None:
			allowed, error = False, f"Conversation {conversation_id} was not found."
		if not message.strip():
			allowed, error = False, "Notification message cannot be empty."
		if not allowed:
			return _finish_action(
				db,
				tool_name="send_notification",
				conversation_id=conversation_id,
				args=args,
				success=False,
				data={},
				error=error,
			)
		if conversation is not None:
			db.add(
				Message(
					conversation_id=conversation.id,
					role="system",
					content=message,
					channel=channel,
				)
			)
		data = {
			"customer_id": customer.id,
			"conversation_id": conversation_id,
			"channel": channel,
			"sent": True,
		}
		return _finish_action(
			db,
			tool_name="send_notification",
			conversation_id=conversation_id,
			args=args,
			success=True,
			data=data,
			error=None,
		)