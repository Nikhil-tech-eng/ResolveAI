"""Build customer context exclusively from deterministic read tools."""

import json
import logging

from sqlalchemy.orm import Session

from agents.schemas import (
	ContextSummary,
	CustomerContext,
	CustomerProfile,
	RecentOrder,
	SubscriptionSummary,
)
from llm.client import call_llm_json
from tools import read_tools

CONTEXT_SUMMARY_PROMPT = """Write one concise paragraph summarizing the supplied
customer support context. Use only the supplied database facts. Do not infer or
invent order, payment, subscription, or account details."""

logger = logging.getLogger(__name__)


def _summary_fallback(
	profile: CustomerProfile,
	orders: list[RecentOrder],
	past_ticket_count: int,
	subscription: SubscriptionSummary,
) -> str:
	"""Create a deterministic one-paragraph summary from loaded database facts."""
	latest_order = orders[0] if orders else None
	order_summary = (
		f"The most recent order is {latest_order.order_id} with status "
		f"{latest_order.status} and payment status {latest_order.payment_status}."
		if latest_order
		else "No recent orders are recorded."
	)
	return (
		f"{profile.name} is a {subscription.tier} customer with account status "
		f"{profile.account_status} and subscription status {subscription.status}. "
		f"There are {past_ticket_count} past support tickets. {order_summary}"
	)


def build_context(customer_id: str, session: Session | None = None) -> CustomerContext:
	"""Load all context through read tools and ask Gemini only to summarize it."""
	customer_result = read_tools.get_customer(customer_id, session=session)
	if not customer_result["success"]:
		raise ValueError(customer_result["error"] or f"Customer {customer_id} was not found.")
	order_result = read_tools.get_customer_orders(customer_id, session=session)
	if not order_result["success"]:
		raise ValueError(order_result["error"] or "Could not load customer orders.")
	ticket_result = read_tools.get_ticket_history(customer_id, session=session)
	if not ticket_result["success"]:
		raise ValueError(ticket_result["error"] or "Could not load ticket history.")
	subscription_result = read_tools.get_subscription(customer_id, session=session)
	if not subscription_result["success"]:
		raise ValueError(subscription_result["error"] or "Could not load subscription.")

	profile_data = customer_result["data"]
	profile = CustomerProfile(
		customer_id=profile_data["id"],
		name=profile_data["name"],
		email=profile_data["email"],
		tier=profile_data["tier"],
		phone=profile_data["phone"],
		account_status=profile_data["account_status"],
		failed_login_attempts=profile_data["failed_login_attempts"],
		created_at=profile_data["created_at"],
	)
	orders = [RecentOrder.model_validate(order) for order in order_result["data"]["orders"]]
	ticket_count = len(ticket_result["data"]["tickets"])
	subscription_data = subscription_result["data"]
	subscription = SubscriptionSummary(
		status=subscription_data["subscription_status"],
		tier=subscription_data["tier"],
	)
	fallback = _summary_fallback(profile, orders, ticket_count, subscription)
	facts = {
		"profile": {
			"customer_id": profile.customer_id,
			"name": profile.name,
			"tier": profile.tier,
			"account_status": profile.account_status,
			"failed_login_attempts": profile.failed_login_attempts,
		},
		"recent_orders": [
			{
				"order_id": order.order_id,
				"status": order.status,
				"total": order.total,
				"payment_status": order.payment_status,
			}
			for order in orders
		],
		"past_tickets_count": ticket_count,
		"subscription": subscription.model_dump(mode="json"),
	}
	try:
		summary_result = call_llm_json(
			CONTEXT_SUMMARY_PROMPT,
			json.dumps(facts, ensure_ascii=True),
			ContextSummary,
			"context",
		)
		summary = summary_result.summary
	except Exception as error:
		logger.warning("Context summary failed; using database-grounded template: %s", error)
		summary = fallback
	return CustomerContext(
		profile=profile,
		tier=subscription.tier,
		recent_orders=orders,
		past_tickets_count=ticket_count,
		subscription=subscription,
		summary=summary,
	)