"""Tests for deterministic read tools, policies, and logged actions."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from data.seed_data import seed_database
from db.models import ActionLog, Base, Conversation, Customer, Message, Order
from tools.action_tools import (
	cancel_order,
	create_return_request,
	generate_return_label,
	initiate_refund,
	reset_password,
	schedule_replacement,
	send_notification,
	unlock_account,
)
from tools.policy import (
	check_cancel_policy,
	check_password_reset_policy,
	check_refund_policy,
	check_return_policy,
	check_unlock_account_policy,
	return_shipping_fee,
)
from tools.read_tools import (
	get_customer,
	get_order,
	get_order_items,
	get_payment_logs,
	get_shipment,
	get_subscription,
	get_ticket_history,
)


@pytest.fixture(scope="module")
def db_session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Session]:
	"""Provide one isolated database containing the phase 2 seed records."""
	database_path: Path = tmp_path_factory.mktemp("tool-data") / "support.db"
	database_url = f"sqlite:///{database_path.as_posix()}"
	seed_database(database_url)
	engine = create_engine(database_url, connect_args={"check_same_thread": False})
	with Session(engine) as session:
		yield session
	engine.dispose()


def test_all_read_tools_return_structured_data(db_session: Session) -> None:
	"""Verify every required read tool returns customer/order facts."""
	results = (
		get_customer("C001", db_session),
		get_order("ORD10001", db_session),
		get_order_items("ORD10001", db_session),
		get_shipment("ORD10001", db_session),
		get_payment_logs("ORD10002", db_session),
		get_ticket_history("C001", db_session),
		get_subscription("C004", db_session),
	)
	assert all(result["success"] for result in results)
	assert all(set(result) == {"success", "data", "error"} for result in results)
	assert results[0]["data"]["id"] == "C001"
	assert results[2]["data"]["items"][0]["sku"] == "SKU-HEAD-1001"
	assert results[5]["data"]["tickets"]
	assert results[6]["data"]["subscription_status"] == "paid_not_activated_sync_failed"


@pytest.mark.parametrize(
	("tool", "identifier"),
	[
		(get_customer, "NO-CUSTOMER"),
		(get_order, "NO-ORDER"),
		(get_order_items, "NO-ORDER"),
		(get_shipment, "NO-ORDER"),
		(get_payment_logs, "NO-ORDER"),
		(get_ticket_history, "NO-CUSTOMER"),
		(get_subscription, "NO-CUSTOMER"),
	],
)
def test_read_tools_return_clear_errors_for_unknown_ids(
	db_session: Session, tool, identifier: str
) -> None:
	"""Verify missing customer/order IDs never produce invented facts."""
	result = tool(identifier, db_session)
	assert result["success"] is False
	assert result["data"] == {}
	assert result["error"]


def test_return_inside_window_is_allowed_and_logged(db_session: Session) -> None:
	"""Verify a delivered in-window order can start a return."""
	result = create_return_request("ORD10001", "CONV-C001", db_session)
	assert result["success"] is True
	assert result["data"]["return_request_id"] == "RET-ORD10001"
	assert result["data"]["return_shipping_fee"] == 0.0
	log = db_session.scalar(
		select(ActionLog).where(ActionLog.tool_name == "create_return_request")
	)
	assert log is not None and log.success is True


def test_c007_return_is_blocked_outside_thirty_day_window(db_session: Session) -> None:
	"""Verify C007's 45-day-old delivery is outside the return window."""
	result = create_return_request("ORD10007", "CONV-C007", db_session)
	assert result["success"] is False
	assert "45 days since delivery" in result["error"]
	assert "within 30 days" in result["error"]


def test_return_label_obeys_policy_and_premium_shipping(db_session: Session) -> None:
	"""Verify premium customers receive a free eligible return label."""
	result = generate_return_label("ORD10001", "CONV-C001", db_session)
	assert result["success"] is True
	assert result["data"]["shipping_fee"] == 0.0
	blocked = generate_return_label("ORD10007", "CONV-C007", db_session)
	assert blocked["success"] is False
	assert "30 days" in blocked["error"]


def test_standard_customer_return_shipping_has_fee(db_session: Session) -> None:
	"""Verify standard customers do not receive the premium shipping waiver."""
	customer = db_session.get(Customer, "C002")
	assert customer is not None
	assert return_shipping_fee(customer) == 5.99


def test_refund_policy_allows_paid_order(db_session: Session) -> None:
	"""Verify a paid order meets the refund policy."""
	order = db_session.get(Order, "ORD10001")
	assert order is not None
	assert check_refund_policy(order)[0] is True


def test_refund_policy_allows_failed_payment_when_amount_was_deducted(
	db_session: Session,
) -> None:
	"""Verify deducted failed payments can be refunded."""
	order = db_session.get(Order, "ORD10002")
	assert order is not None
	result = initiate_refund("ORD10002", "CONV-C002", db_session)
	assert result["success"] is True
	assert order.payment_status == "refunded"


def test_refund_policy_blocks_unpaid_undeducted_order() -> None:
	"""Verify failed payments without a deduction are not refundable."""
	order = SimpleNamespace(id="ORD-TEST", payment_status="failed")
	payment = SimpleNamespace(amount_deducted=False)
	allowed, error = check_refund_policy(order, [payment])
	assert allowed is False
	assert "no payment amount was deducted" in error


def test_cancel_placed_order_succeeds_and_is_logged(db_session: Session) -> None:
	"""Verify a placed order can be cancelled."""
	result = cancel_order("ORD10006", "CONV-C006", db_session)
	assert result["success"] is True
	assert result["data"]["status"] == "cancelled"
	assert db_session.scalar(
		select(ActionLog).where(
			ActionLog.tool_name == "cancel_order",
			ActionLog.success.is_(True),
		)
	)


def test_cancel_shipped_order_fails_with_policy_error_and_log(db_session: Session) -> None:
	"""Verify a shipped order is blocked with a clear reason and action log."""
	result = cancel_order("ORD10005", "CONV-C005", db_session)
	assert result["success"] is False
	assert "only orders with status 'placed'" in result["error"]
	assert "current status: shipped" in result["error"]
	log = db_session.scalar(
		select(ActionLog).where(
			ActionLog.tool_name == "cancel_order",
			ActionLog.conversation_id == "CONV-C005",
			ActionLog.success.is_(False),
		)
	)
	assert log is not None
	assert any(
		entry.result["error"] == result["error"]
		for entry in db_session.scalars(
			select(ActionLog).where(
				ActionLog.tool_name == "cancel_order",
				ActionLog.conversation_id == "CONV-C005",
				ActionLog.success.is_(False),
			)
		)
	)


def test_reset_password_for_existing_customer_is_logged(db_session: Session) -> None:
	"""Verify a password reset starts for an existing non-suspended customer."""
	result = reset_password("C001", "CONV-C001", db_session)
	assert result["success"] is True
	assert result["data"]["reset_link_sent"] is True


def test_password_reset_policy_blocks_suspended_customer(db_session: Session) -> None:
	"""Verify suspended accounts cannot request password resets."""
	customer = SimpleNamespace(account_status="suspended")
	allowed, error = check_password_reset_policy(customer)
	assert allowed is False
	assert "suspended" in error


def test_locked_account_can_be_unlocked(db_session: Session) -> None:
	"""Verify unlocking C003 clears the lock and failed-attempt count."""
	result = unlock_account("C003", "CONV-C003", db_session)
	assert result["success"] is True
	customer = db_session.get(Customer, "C003")
	assert customer is not None
	assert customer.account_status == "active"
	assert customer.failed_login_attempts == 0


def test_active_account_cannot_be_unlocked(db_session: Session) -> None:
	"""Verify the unlock policy rejects accounts that are not locked."""
	customer = db_session.get(Customer, "C001")
	assert customer is not None
	allowed, error = check_unlock_account_policy(customer)
	assert allowed is False
	assert "status is 'active'" in error


def test_verified_wrong_item_can_be_replaced(db_session: Session) -> None:
	"""Verify the mismatched C001 shipment automatically verifies a wrong item."""
	result = schedule_replacement("ORD10001", conversation_id="CONV-C001", session=db_session)
	assert result["success"] is True
	assert result["data"]["verification"] == "wrong_item"


def test_unverified_replacement_is_blocked_and_logged(db_session: Session) -> None:
	"""Verify an unverified replacement request is denied and logged."""
	result = schedule_replacement("ORD10010", conversation_id="CONV-C008", session=db_session)
	assert result["success"] is False
	assert "until a wrong or damaged item is verified" in result["error"]
	logs = db_session.scalars(
		select(ActionLog).where(
			ActionLog.tool_name == "schedule_replacement",
			ActionLog.conversation_id == "CONV-C008",
		)
	).all()
	assert any(log.success is False and log.args["verified"] is False for log in logs)


def test_explicitly_verified_damaged_item_can_be_replaced(db_session: Session) -> None:
	"""Verify an explicitly verified damaged-item claim may be replaced."""
	result = schedule_replacement(
		"ORD10010", verified=True, conversation_id="CONV-C008", session=db_session
	)
	assert result["success"] is True
	assert result["data"]["verification"] == "customer_verified"


def test_send_notification_records_message_and_action(db_session: Session) -> None:
	"""Verify notifications tied to a conversation are persisted and logged."""
	result = send_notification(
		"C001", "Your return request is being reviewed.", "CONV-C001", session=db_session
	)
	assert result["success"] is True
	message = db_session.scalar(
		select(Message).where(
			Message.conversation_id == "CONV-C001",
			Message.content == "Your return request is being reviewed.",
		)
	)
	assert message is not None and message.role == "system"
	log = db_session.scalar(
		select(ActionLog).where(ActionLog.tool_name == "send_notification")
	)
	assert log is not None and log.success is True


def test_send_notification_rejects_mismatched_customer_conversation(
	db_session: Session,
) -> None:
	"""Verify a notification cannot be logged against another customer's conversation."""
	result = send_notification(
		"C002", "Order update", "CONV-C001", session=db_session
	)
	assert result["success"] is False
	assert "does not belong" in result["error"]


def test_missing_order_action_is_still_logged(db_session: Session) -> None:
	"""Verify failed action attempts against unknown orders also create action logs."""
	result = cancel_order("MISSING-ORDER", session=db_session)
	assert result["success"] is False
	assert "was not found" in result["error"]
	assert db_session.scalar(
		select(ActionLog).where(
			ActionLog.tool_name == "cancel_order",
			ActionLog.args["order_id"].as_string() == "MISSING-ORDER",
		)
	)