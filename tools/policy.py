"""Deterministic policy checks for support actions."""

from datetime import datetime, timedelta, timezone
from typing import Sequence

from db.models import Conversation, Customer, Order, Payment

PolicyDecision = tuple[bool, str | None]
STANDARD_RETURN_SHIPPING_FEE = 5.99
RETURN_WINDOW = timedelta(days=30)


def _to_utc(value: datetime) -> datetime:
	"""Normalize naive or aware datetimes to timezone-aware UTC."""
	if value.tzinfo is None:
		return value.replace(tzinfo=timezone.utc)
	return value.astimezone(timezone.utc)


def check_return_policy(
	order: Order | None,
	*,
	delivered_at: datetime | None = None,
	now: datetime | None = None,
) -> PolicyDecision:
	"""Allow returns only for delivered orders within 30 days of delivery."""
	if order is None:
		return False, "Order was not found."
	if order.status != "delivered":
		return False, f"A return requires a delivered order (current status: {order.status})."

	delivery_time = delivered_at or order.delivered_at
	if delivery_time is None:
		return False, "A return cannot be checked because the delivery date is unavailable."

	current_time = _to_utc(now or datetime.now(timezone.utc))
	delivery_time = _to_utc(delivery_time)
	if delivery_time > current_time:
		return False, "A return cannot be started before the order is delivered."
	if current_time - delivery_time > RETURN_WINDOW:
		days_since_delivery = (current_time - delivery_time).days
		return (
			False,
			f"Return window expired: {days_since_delivery} days since delivery; "
			"returns are allowed within 30 days.",
		)
	return True, None


def check_refund_policy(
	order: Order | None, payments: Sequence[Payment] = ()
) -> PolicyDecision:
	"""Allow refunds for paid orders or when a payment was actually deducted."""
	if order is None:
		return False, "Order was not found."
	if order.payment_status == "paid" or any(payment.amount_deducted for payment in payments):
		return True, None
	return (
		False,
		"Refund is blocked: the order is not paid and no payment amount was deducted.",
	)


def check_cancel_policy(order: Order | None) -> PolicyDecision:
	"""Allow cancellation only while an order is still placed."""
	if order is None:
		return False, "Order was not found."
	if order.status != "placed":
		return (
			False,
			f"Cannot cancel order {order.id}: only orders with status 'placed' "
			f"may be cancelled (current status: {order.status}).",
		)
	return True, None


def check_replacement_policy(verified: bool) -> PolicyDecision:
	"""Allow replacement only after a wrong or damaged item is verified."""
	if not verified:
		return False, "Replacement is blocked until a wrong or damaged item is verified."
	return True, None


def return_shipping_fee(customer: Customer | None) -> float:
	"""Return the applicable fee, with free return shipping for premium customers."""
	if customer is not None and customer.tier == "premium":
		return 0.0
	return STANDARD_RETURN_SHIPPING_FEE


def check_password_reset_policy(customer: Customer | None) -> PolicyDecision:
	"""Allow password resets for existing accounts that are not suspended."""
	if customer is None:
		return False, "Customer was not found."
	if customer.account_status == "suspended":
		return False, "Password reset is blocked for suspended accounts."
	return True, None


def check_unlock_account_policy(customer: Customer | None) -> PolicyDecision:
	"""Allow account unlocking only when the account is currently locked."""
	if customer is None:
		return False, "Customer was not found."
	if customer.account_status != "locked":
		return False, f"Account cannot be unlocked while status is '{customer.account_status}'."
	return True, None


def check_notification_policy(
	customer: Customer | None, conversation: Conversation | None = None
) -> PolicyDecision:
	"""Require an existing customer and a matching conversation if supplied."""
	if customer is None:
		return False, "Customer was not found."
	if conversation is not None and conversation.customer_id != customer.id:
		return False, "Conversation does not belong to the specified customer."
	return True, None