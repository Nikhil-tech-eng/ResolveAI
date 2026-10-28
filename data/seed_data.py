"""Create and populate the local support database with demo scenarios."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import (
	ActionLog,
	Base,
	CSAT,
	Conversation,
	Customer,
	Message,
	Order,
	OrderItem,
	Payment,
	Shipment,
	Ticket,
)
from db.session import DATABASE_PATH, DATABASE_URL

logger = logging.getLogger(__name__)

FILLER_CUSTOMERS = (
	("C011", "Riley Adams"),
	("C012", "Parker Shah"),
	("C013", "Quinn Davis"),
	("C014", "Avery Thompson"),
	("C015", "Emerson Garcia"),
	("C016", "Finley Martin"),
	("C017", "Rowan Wilson"),
	("C018", "Sage Robinson"),
	("C019", "Dakota Lewis"),
	("C020", "Reese Walker"),
)

SCENARIO_PROFILES = (
	("C001", "Ava Patel", "premium", "active", 0, "active"),
	("C002", "Jordan Lee", "standard", "active", 0, "active"),
	("C003", "Morgan Chen", "standard", "locked", 5, "active"),
	("C004", "Olivia Martin", "premium", "active", 0, "paid_not_activated_sync_failed"),
	("C005", "Sam Rivera", "standard", "active", 0, "active"),
	("C006", "Taylor Brooks", "standard", "active", 0, "active"),
	("C007", "Casey Nguyen", "standard", "active", 0, "active"),
	("C008", "Jamie Clark", "standard", "active", 0, "active"),
	("C009", "Alex Kim", "standard", "active", 0, "active"),
	("C010", "Drew Wilson", "standard", "active", 0, "active"),
)

PRODUCTS = (
	("SKU-CABLE-01", "USB-C Charging Cable", 14.99),
	("SKU-MUG-02", "Ceramic Travel Mug", 24.50),
	("SKU-LAMP-03", "Desk Lamp", 39.00),
	("SKU-BAG-04", "Canvas Tote Bag", 18.75),
	("SKU-SPEAKER-05", "Bluetooth Speaker", 64.99),
)


def _add_order(
	session: Session,
	*,
	order_id: str,
	customer_id: str,
	status: str,
	payment_status: str,
	placed_at: datetime,
	delivered_at: datetime | None,
	sku: str,
	product_name: str,
	price: float,
	payment_amount_deducted: bool,
	shipment_status: str | None = None,
	delivered_item_sku: str | None = None,
) -> None:
	"""Add an order and its item, payment, and applicable shipment records."""
	session.add(
		Order(
			id=order_id,
			customer_id=customer_id,
			status=status,
			total=price,
			payment_status=payment_status,
			placed_at=placed_at,
			delivered_at=delivered_at,
		)
	)
	session.add(
		OrderItem(order_id=order_id, sku=sku, name=product_name, qty=1, price=price)
	)
	session.add(
		Payment(
			order_id=order_id,
			amount=price,
			status=payment_status,
			gateway_ref=f"GW-{order_id}",
			amount_deducted=payment_amount_deducted,
			created_at=placed_at,
		)
	)
	if status in {"shipped", "delivered", "returned"}:
		resolved_shipment_status = shipment_status or status
		session.add(
			Shipment(
				order_id=order_id,
				carrier="ParcelPro" if int(order_id[-1]) % 2 else "Northstar Shipping",
				tracking_no=f"TRK-{order_id}",
				status=resolved_shipment_status,
				delivered_item_sku=delivered_item_sku or sku,
				delivered_at=delivered_at,
			)
		)


def _add_scenario_conversations(session: Session, now: datetime) -> None:
	"""Add scenario conversations and representative customer/agent messages."""
	scenarios = (
		("C001", "wrong_item", "The headphones delivered are not the item I ordered.", {"order_id": "ORD10001"}),
		("C002", "payment_failure", "My payment failed, but the amount was deducted.", {"order_id": "ORD10002"}),
		("C003", "login_issue", "I cannot sign in to my locked account.", {"account_status": "locked"}),
		("C004", "subscription_issue", "My subscription payment went through, but it is not active.", {"sync_failed": True}),
		("C005", "cancel_order", "Please cancel ORD10005 even though it has shipped.", {"order_id": "ORD10005", "cancel_policy": "blocked"}),
		("C006", "cancel_order", "Please cancel ORD10006 before it ships.", {"order_id": "ORD10006", "cancel_policy": "allowed"}),
		("C007", "refund_request", "I want to return my order delivered 45 days ago.", {"order_id": "ORD10007", "escalate": True}),
		("C008", "fraud_complaint", "I see charges I do not recognize.", {"order_id": "ORD10010"}),
		("C009", "order_status", "This order is very late and I am extremely frustrated.", {"order_id": "ORD10008", "sentiment": "strongly_negative"}),
		("C010", "order_status", "Can you tell me the status of my order?", {"order_id": "ORD10009"}),
	)
	for index, (customer_id, intent, content, state) in enumerate(scenarios):
		conversation_id = f"CONV-{customer_id}"
		session.add(
			Conversation(
				id=conversation_id,
				customer_id=customer_id,
				channel="web",
				state={"intent": intent, "scenario": customer_id, **state},
				status="open",
				started_at=now - timedelta(hours=2, minutes=index),
			)
		)
		session.add_all(
			[
				Message(
					conversation_id=conversation_id,
					role="customer",
					content=content,
					channel="web",
					timestamp=now - timedelta(hours=2, minutes=index),
				),
				Message(
					conversation_id=conversation_id,
					role="agent",
					content="I am checking the relevant account and order details.",
					channel="web",
					timestamp=now - timedelta(hours=2, minutes=index) + timedelta(seconds=30),
				),
			]
		)


def _add_payment_failure_spike(session: Session, now: datetime) -> None:
	"""Add 35 payment-failure conversations within a 35-minute interval."""
	start = now - timedelta(minutes=45)
	for index in range(35):
		customer_id, _ = FILLER_CUSTOMERS[index % len(FILLER_CUSTOMERS)]
		timestamp = start + timedelta(minutes=index)
		conversation_id = f"CONV-PAYFAIL-{index + 1:03d}"
		session.add(
			Conversation(
				id=conversation_id,
				customer_id=customer_id,
				channel="web",
				state={"intent": "payment_failure", "source": "root_cause_spike"},
				status="open",
				started_at=timestamp,
			)
		)
		session.add(
			Message(
				conversation_id=conversation_id,
				role="customer",
				content="Checkout reports payment failed, but my bank shows a deduction.",
				channel="web",
				timestamp=timestamp,
			)
		)


def _populate_database(session: Session, now: datetime) -> None:
	"""Populate all customer, order, ticket, conversation, and feedback records."""
	for index, profile in enumerate(SCENARIO_PROFILES):
		customer_id, name, tier, account_status, failed_attempts, subscription = profile
		session.add(
			Customer(
				id=customer_id,
				name=name,
				email=f"{customer_id.lower()}@example.com",
				phone=f"+1-555-010-{index:02d}",
				tier=tier,
				account_status=account_status,
				failed_login_attempts=failed_attempts,
				subscription_status=subscription,
				created_at=now - timedelta(days=365 - index * 11),
			)
		)
	for index, (customer_id, name) in enumerate(FILLER_CUSTOMERS):
		session.add(
			Customer(
				id=customer_id,
				name=name,
				email=f"{customer_id.lower()}@example.com",
				phone=f"+1-555-011-{index:02d}",
				tier="premium" if index % 4 == 0 else "standard",
				account_status="active",
				failed_login_attempts=0,
				subscription_status="active" if index % 3 == 0 else "none",
				created_at=now - timedelta(days=180 - index * 7),
			)
		)
	session.flush()

	scenario_orders = (
		("ORD10001", "C001", "delivered", "paid", 89.99, 20, 15, "SKU-HEAD-1001", "Wireless Headphones", "SKU-CASE-9001", None, True),
		("ORD10002", "C002", "placed", "failed", 42.50, 3, None, "SKU-KETTLE-02", "Electric Kettle", None, None, True),
		("ORD10003", "C003", "delivered", "paid", 32.00, 40, 34, "SKU-BOOK-03", "Hardcover Notebook", None, None, True),
		("ORD10004", "C004", "placed", "paid", 59.00, 5, None, "SKU-SKIN-04", "Skincare Set", None, None, True),
		("ORD10005", "C005", "shipped", "paid", 75.25, 2, None, "SKU-SHOE-05", "Running Shoes", None, None, True),
		("ORD10006", "C006", "placed", "paid", 27.99, 1, None, "SKU-CASE-06", "Phone Case", None, None, True),
		("ORD10007", "C007", "delivered", "paid", 115.00, 60, 45, "SKU-COAT-07", "Rain Jacket", None, None, True),
		("ORD10008", "C009", "shipped", "paid", 149.99, 12, None, "SKU-CHAIR-08", "Office Chair", None, "delayed", True),
		("ORD10009", "C010", "shipped", "paid", 21.50, 1, None, "SKU-BOTTLE-09", "Water Bottle", None, None, True),
		("ORD10010", "C008", "delivered", "paid", 19.95, 7, 3, "SKU-CHARGER-10", "Wall Charger", None, None, True),
	)
	for (
		order_id,
		customer_id,
		status,
		payment_status,
		price,
		days_ago,
		delivered_days_ago,
		sku,
		product_name,
		delivered_sku,
		shipment_status,
		amount_deducted,
	) in scenario_orders:
		_add_order(
			session,
			order_id=order_id,
			customer_id=customer_id,
			status=status,
			payment_status=payment_status,
			placed_at=now - timedelta(days=days_ago),
			delivered_at=(
				now - timedelta(days=delivered_days_ago)
				if delivered_days_ago is not None
				else None
			),
			sku=sku,
			product_name=product_name,
			price=price,
			payment_amount_deducted=amount_deducted,
			shipment_status=shipment_status,
			delivered_item_sku=delivered_sku,
		)

	for number in range(10011, 10051):
		customer_index = (number - 10011) % len(FILLER_CUSTOMERS)
		customer_id = FILLER_CUSTOMERS[customer_index][0]
		sku, product_name, price = PRODUCTS[(number - 10011) % len(PRODUCTS)]
		status = ("delivered", "shipped", "placed", "delivered")[number % 4]
		payment_status = "paid" if number % 9 else "pending"
		days_ago = 2 + (number % 50)
		delivered_days_ago = days_ago - 3 if status == "delivered" else None
		_add_order(
			session,
			order_id=f"ORD{number}",
			customer_id=customer_id,
			status=status,
			payment_status=payment_status,
			placed_at=now - timedelta(days=days_ago),
			delivered_at=(
				now - timedelta(days=delivered_days_ago)
				if delivered_days_ago is not None
				else None
			),
			sku=sku,
			product_name=product_name,
			price=price,
			payment_amount_deducted=payment_status == "paid",
		)

	_add_scenario_conversations(session, now)
	_add_payment_failure_spike(session, now)
	session.flush()

	old_tickets = [
		Ticket(
			customer_id="C001",
			order_id="ORD10001",
			intent="general_query",
			category="account_support",
			priority="low",
			sentiment="neutral",
			summary=f"Previous support request {index + 1}.",
			investigation="Customer account details were checked.",
			actions_taken=[{"action": "account_review", "success": True}],
			status="closed" if index == 2 else "resolved",
			created_at=now - timedelta(days=150 - index * 35),
			updated_at=now - timedelta(days=149 - index * 35),
		)
		for index in range(3)
	]
	escalated_ticket = Ticket(
		customer_id="C007",
		order_id="ORD10007",
		intent="refund_request",
		category="returns",
		priority="high",
		sentiment="negative",
		summary="Customer requests a return for an order delivered 45 days ago.",
		investigation="Delivery date is outside the 30-day return window.",
		actions_taken=[],
		status="opened",
		assigned_to="support_queue",
		escalation_reason="Return request is outside the 30-day policy window.",
		sla_deadline=now + timedelta(hours=1),
		created_at=now,
		updated_at=now,
	)
	session.add_all([*old_tickets, escalated_ticket])
	session.flush()

	session.add_all(
		[
			ActionLog(
				conversation_id="CONV-C005",
				tool_name="cancel_order",
				args={"order_id": "ORD10005"},
				result={"success": False, "error": "Order has shipped and cannot be cancelled."},
				success=False,
				timestamp=now,
			),
			ActionLog(
				conversation_id="CONV-C006",
				tool_name="cancel_order",
				args={"order_id": "ORD10006"},
				result={"success": True, "data": {"order_id": "ORD10006", "status": "cancelled"}},
				success=True,
				timestamp=now,
			),
			CSAT(
				ticket_id=old_tickets[0].id,
				conversation_id="CONV-C001",
				rating=5,
				sentiment="positive",
				resolution_quality="resolved",
				comment="The previous issue was handled quickly.",
				created_at=now - timedelta(days=149),
			),
		]
	)


def seed_database(database_url: str | None = None) -> None:
	"""Recreate and populate the support database at the requested SQLite URL."""
	selected_url = database_url or DATABASE_URL
	if selected_url.startswith("sqlite:///"):
		database_path = Path(selected_url.removeprefix("sqlite:///"))
		database_path.parent.mkdir(parents=True, exist_ok=True)
	engine = create_engine(
		selected_url,
		connect_args={"check_same_thread": False}
		if selected_url.startswith("sqlite")
		else {},
	)
	try:
		Base.metadata.drop_all(bind=engine)
		Base.metadata.create_all(bind=engine)
		factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
		with factory() as session:
			_populate_database(session, datetime.now(timezone.utc))
			session.commit()
	finally:
		engine.dispose()
	logger.info("Seeded support database at %s", database_path if selected_url == DATABASE_URL else selected_url)


def main() -> None:
	"""Seed the default project database when run as a script."""
	logging.basicConfig(level=logging.INFO)
	seed_database()
	print(f"Seeded database: {DATABASE_PATH}")


if __name__ == "__main__":
	main()