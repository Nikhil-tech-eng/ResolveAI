"""Verify the required section 12 data is present in the generated database."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from data.seed_data import seed_database
from db.models import (
    ActionLog,
    Base,
    Conversation,
    Customer,
    Message,
    Order,
    OrderItem,
    Payment,
    Shipment,
    Ticket,
)


@pytest.fixture(scope="module")
def seeded_session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Session]:
    """Create one isolated seeded SQLite database for this test module."""
    database_path = tmp_path_factory.mktemp("seed-data") / "support.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    seed_database(database_url)
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_all_ten_required_scenarios_are_seeded(seeded_session: Session) -> None:
    """Verify each required customer and scenario-specific record."""
    session = seeded_session

    customer = session.get(Customer, "C001")
    order = session.get(Order, "ORD10001")
    assert customer is not None and customer.tier == "premium"
    assert session.scalar(
        select(func.count()).select_from(Ticket).where(Ticket.customer_id == "C001")
    ) == 3
    assert order is not None and order.status == "delivered"
    item = session.scalar(select(OrderItem).where(OrderItem.order_id == "ORD10001"))
    shipment = session.scalar(select(Shipment).where(Shipment.order_id == "ORD10001"))
    assert item is not None and shipment is not None
    assert shipment.delivered_item_sku != item.sku

    order = session.get(Order, "ORD10002")
    payment = session.scalar(select(Payment).where(Payment.order_id == "ORD10002"))
    assert order is not None and order.payment_status == "failed"
    assert payment is not None and payment.status == "failed" and payment.amount_deducted

    customer = session.get(Customer, "C003")
    assert customer is not None
    assert customer.account_status == "locked" and customer.failed_login_attempts == 5

    customer = session.get(Customer, "C004")
    conversation = session.get(Conversation, "CONV-C004")
    assert customer is not None and customer.tier == "premium"
    assert customer.subscription_status == "paid_not_activated_sync_failed"
    assert conversation is not None and conversation.state["sync_failed"] is True

    order = session.get(Order, "ORD10005")
    blocked_conversation = session.get(Conversation, "CONV-C005")
    assert order is not None and order.status == "shipped"
    assert blocked_conversation is not None
    assert blocked_conversation.state["cancel_policy"] == "blocked"

    order = session.get(Order, "ORD10006")
    allowed_conversation = session.get(Conversation, "CONV-C006")
    assert order is not None and order.status == "placed"
    assert allowed_conversation is not None
    assert allowed_conversation.state["cancel_policy"] == "allowed"
    assert session.scalar(
        select(ActionLog).where(ActionLog.conversation_id == "CONV-C006")
    ).success

    order = session.get(Order, "ORD10007")
    assert order is not None and order.delivered_at is not None
    days_since_delivery = (
        datetime.now(timezone.utc).replace(tzinfo=None) - order.delivered_at
    ).days
    assert 44 <= days_since_delivery <= 46
    ticket = session.scalar(select(Ticket).where(Ticket.customer_id == "C007"))
    assert ticket is not None and "outside the 30-day" in ticket.escalation_reason

    fraud_conversation = session.get(Conversation, "CONV-C008")
    assert fraud_conversation is not None
    assert fraud_conversation.state["intent"] == "fraud_complaint"

    delayed_order = session.get(Order, "ORD10008")
    delayed_shipment = session.scalar(
        select(Shipment).where(Shipment.order_id == "ORD10008")
    )
    angry_conversation = session.get(Conversation, "CONV-C009")
    assert delayed_order is not None and delayed_order.status == "shipped"
    assert delayed_shipment is not None and delayed_shipment.status == "delayed"
    assert angry_conversation is not None
    assert angry_conversation.state["sentiment"] == "strongly_negative"

    status_conversation = session.get(Conversation, "CONV-C010")
    assert status_conversation is not None
    assert status_conversation.state["intent"] == "order_status"
    assert status_conversation.state["order_id"] == "ORD10009"


def test_seed_has_filler_customers_and_about_fifty_orders(seeded_session: Session) -> None:
    """Verify the seed includes the requested filler population and order volume."""
    assert seeded_session.scalar(select(func.count()).select_from(Customer)) >= 20
    assert seeded_session.scalar(select(func.count()).select_from(Order)) == 50


def test_payment_failure_conversations_are_clustered_within_one_hour(
    seeded_session: Session,
) -> None:
    """Verify the root-cause sample has at least 30 clustered conversations/messages."""
    conversations = [
        conversation
        for conversation in seeded_session.scalars(select(Conversation))
        if conversation.state.get("intent") == "payment_failure"
        and conversation.state.get("source") == "root_cause_spike"
    ]
    assert len(conversations) >= 30
    start_times = [conversation.started_at for conversation in conversations]
    assert max(start_times) - min(start_times) <= timedelta(hours=1)

    conversation_ids = [conversation.id for conversation in conversations]
    message_count = seeded_session.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id.in_(conversation_ids))
    )
    assert message_count >= 30