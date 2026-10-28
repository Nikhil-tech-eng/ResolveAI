"""Read-only database tools for customer and order context."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import (
	Conversation,
	Customer,
	Order,
	OrderItem,
	Payment,
	Shipment,
	Ticket,
)
from db.session import SessionLocal


class ToolResult(TypedDict):
	"""Common result shape returned by support tools."""

	success: bool
	data: dict[str, Any]
	error: str | None


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a caller session or open a short-lived session for one read."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _success(data: dict[str, Any]) -> ToolResult:
	"""Build a successful tool result."""
	return {"success": True, "data": data, "error": None}


def _failure(error: str) -> ToolResult:
	"""Build a failed tool result."""
	return {"success": False, "data": {}, "error": error}


def _iso(value: datetime | None) -> str | None:
	"""Serialize an optional datetime as ISO-8601 text."""
	return value.isoformat() if value is not None else None


def get_customer(customer_id: str, session: Session | None = None) -> ToolResult:
	"""Return a customer's profile and account state."""
	with _using_session(session) as db:
		customer = db.get(Customer, customer_id)
		if customer is None:
			return _failure(f"Customer {customer_id} was not found.")
		return _success(
			{
				"id": customer.id,
				"name": customer.name,
				"email": customer.email,
				"phone": customer.phone,
				"tier": customer.tier,
				"account_status": customer.account_status,
				"failed_login_attempts": customer.failed_login_attempts,
				"subscription_status": customer.subscription_status,
				"created_at": _iso(customer.created_at),
			}
		)


def get_order(order_id: str, session: Session | None = None) -> ToolResult:
	"""Return an order's identity, status, totals, and customer ID."""
	with _using_session(session) as db:
		order = db.get(Order, order_id)
		if order is None:
			return _failure(f"Order {order_id} was not found.")
		return _success(
			{
				"id": order.id,
				"customer_id": order.customer_id,
				"status": order.status,
				"total": order.total,
				"payment_status": order.payment_status,
				"placed_at": _iso(order.placed_at),
				"delivered_at": _iso(order.delivered_at),
			}
		)


def get_customer_orders(
	customer_id: str,
	limit: int = 5,
	session: Session | None = None,
) -> ToolResult:
	"""Return a customer's most recent orders using database-backed facts."""
	if limit <= 0:
		return _failure("Order limit must be a positive integer.")
	with _using_session(session) as db:
		if db.get(Customer, customer_id) is None:
			return _failure(f"Customer {customer_id} was not found.")
		orders = db.scalars(
			select(Order)
			.where(Order.customer_id == customer_id)
			.order_by(Order.placed_at.desc(), Order.id.desc())
			.limit(limit)
		).all()
		return _success(
			{
				"customer_id": customer_id,
				"orders": [
					{
						"order_id": order.id,
						"status": order.status,
						"total": order.total,
						"payment_status": order.payment_status,
						"placed_at": _iso(order.placed_at),
						"delivered_at": _iso(order.delivered_at),
					}
					for order in orders
				],
			}
		)


def get_order_items(order_id: str, session: Session | None = None) -> ToolResult:
	"""Return the product lines for an order."""
	with _using_session(session) as db:
		if db.get(Order, order_id) is None:
			return _failure(f"Order {order_id} was not found.")
		items = db.scalars(
			select(OrderItem).where(OrderItem.order_id == order_id).order_by(OrderItem.id)
		).all()
		return _success(
			{
				"order_id": order_id,
				"items": [
					{
						"id": item.id,
						"sku": item.sku,
						"name": item.name,
						"qty": item.qty,
						"price": item.price,
					}
					for item in items
				],
			}
		)


def get_shipment(order_id: str, session: Session | None = None) -> ToolResult:
	"""Return shipment tracking records for an order."""
	with _using_session(session) as db:
		if db.get(Order, order_id) is None:
			return _failure(f"Order {order_id} was not found.")
		shipments = db.scalars(
			select(Shipment)
			.where(Shipment.order_id == order_id)
			.order_by(Shipment.id)
		).all()
		return _success(
			{
				"order_id": order_id,
				"shipments": [
					{
						"id": shipment.id,
						"carrier": shipment.carrier,
						"tracking_no": shipment.tracking_no,
						"status": shipment.status,
						"delivered_item_sku": shipment.delivered_item_sku,
						"delivered_at": _iso(shipment.delivered_at),
					}
					for shipment in shipments
				],
			}
		)


def get_payment_logs(order_id: str, session: Session | None = None) -> ToolResult:
	"""Return payment attempts and deduction flags for an order."""
	with _using_session(session) as db:
		if db.get(Order, order_id) is None:
			return _failure(f"Order {order_id} was not found.")
		payments = db.scalars(
			select(Payment).where(Payment.order_id == order_id).order_by(Payment.id)
		).all()
		return _success(
			{
				"order_id": order_id,
				"payments": [
					{
						"id": payment.id,
						"amount": payment.amount,
						"status": payment.status,
						"gateway_ref": payment.gateway_ref,
						"amount_deducted": payment.amount_deducted,
						"created_at": _iso(payment.created_at),
					}
					for payment in payments
				],
			}
		)


def get_ticket_history(customer_id: str, session: Session | None = None) -> ToolResult:
	"""Return a customer's tickets from newest to oldest."""
	with _using_session(session) as db:
		if db.get(Customer, customer_id) is None:
			return _failure(f"Customer {customer_id} was not found.")
		tickets = db.scalars(
			select(Ticket)
			.where(Ticket.customer_id == customer_id)
			.order_by(Ticket.created_at.desc(), Ticket.id.desc())
		).all()
		return _success(
			{
				"customer_id": customer_id,
				"tickets": [
					{
						"id": ticket.id,
						"order_id": ticket.order_id,
						"intent": ticket.intent,
						"category": ticket.category,
						"priority": ticket.priority,
						"sentiment": ticket.sentiment,
						"summary": ticket.summary,
						"investigation": ticket.investigation,
						"actions_taken": ticket.actions_taken,
						"status": ticket.status,
						"assigned_to": ticket.assigned_to,
						"escalation_reason": ticket.escalation_reason,
						"created_at": _iso(ticket.created_at),
						"updated_at": _iso(ticket.updated_at),
					}
					for ticket in tickets
				],
			}
		)


def get_subscription(customer_id: str, session: Session | None = None) -> ToolResult:
	"""Return the subscription state recorded on a customer's profile."""
	with _using_session(session) as db:
		customer = db.get(Customer, customer_id)
		if customer is None:
			return _failure(f"Customer {customer_id} was not found.")
		return _success(
			{
				"customer_id": customer.id,
				"tier": customer.tier,
				"subscription_status": customer.subscription_status,
			}
		)