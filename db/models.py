"""SQLAlchemy models for the support platform."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
	Boolean,
	CheckConstraint,
	DateTime,
	Float,
	ForeignKey,
	Integer,
	JSON,
	String,
	Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
	"""Return the current timezone-aware UTC timestamp."""
	return datetime.now(timezone.utc)


class Base(DeclarativeBase):
	"""Base class for all support database models."""


class Customer(Base):
	"""Customer profile and account state."""

	__tablename__ = "customers"
	__table_args__ = (
		CheckConstraint("tier IN ('standard', 'premium')", name="ck_customers_tier"),
		CheckConstraint(
			"account_status IN ('active', 'locked', 'suspended')",
			name="ck_customers_account_status",
		),
	)

	id: Mapped[str] = mapped_column(String(20), primary_key=True)
	name: Mapped[str] = mapped_column(String(120), nullable=False)
	email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	phone: Mapped[str | None] = mapped_column(String(30))
	tier: Mapped[str] = mapped_column(String(20), nullable=False, default="standard")
	account_status: Mapped[str] = mapped_column(
		String(20), nullable=False, default="active"
	)
	failed_login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	subscription_status: Mapped[str] = mapped_column(
		String(50), nullable=False, default="none"
	)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)

	orders: Mapped[list["Order"]] = relationship(
		back_populates="customer", cascade="all, delete-orphan"
	)
	tickets: Mapped[list["Ticket"]] = relationship(back_populates="customer")
	conversations: Mapped[list["Conversation"]] = relationship(
		back_populates="customer"
	)


class Order(Base):
	"""Customer order and payment lifecycle state."""

	__tablename__ = "orders"
	__table_args__ = (
		CheckConstraint(
			"status IN ('placed', 'shipped', 'delivered', 'cancelled', 'returned')",
			name="ck_orders_status",
		),
		CheckConstraint(
			"payment_status IN ('paid', 'failed', 'refunded', 'pending')",
			name="ck_orders_payment_status",
		),
	)

	id: Mapped[str] = mapped_column(String(20), primary_key=True)
	customer_id: Mapped[str] = mapped_column(
		ForeignKey("customers.id"), nullable=False, index=True
	)
	status: Mapped[str] = mapped_column(String(20), nullable=False)
	total: Mapped[float] = mapped_column(Float, nullable=False)
	payment_status: Mapped[str] = mapped_column(String(20), nullable=False)
	placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
	delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

	customer: Mapped[Customer] = relationship(back_populates="orders")
	items: Mapped[list["OrderItem"]] = relationship(
		back_populates="order", cascade="all, delete-orphan"
	)
	shipments: Mapped[list["Shipment"]] = relationship(
		back_populates="order", cascade="all, delete-orphan"
	)
	payments: Mapped[list["Payment"]] = relationship(
		back_populates="order", cascade="all, delete-orphan"
	)
	tickets: Mapped[list["Ticket"]] = relationship(back_populates="order")


class OrderItem(Base):
	"""A product line belonging to an order."""

	__tablename__ = "order_items"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	order_id: Mapped[str] = mapped_column(
		ForeignKey("orders.id"), nullable=False, index=True
	)
	sku: Mapped[str] = mapped_column(String(40), nullable=False)
	name: Mapped[str] = mapped_column(String(160), nullable=False)
	qty: Mapped[int] = mapped_column(Integer, nullable=False)
	price: Mapped[float] = mapped_column(Float, nullable=False)

	order: Mapped[Order] = relationship(back_populates="items")


class Shipment(Base):
	"""Shipment tracking and delivered-item details."""

	__tablename__ = "shipments"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	order_id: Mapped[str] = mapped_column(
		ForeignKey("orders.id"), nullable=False, index=True
	)
	carrier: Mapped[str] = mapped_column(String(80), nullable=False)
	tracking_no: Mapped[str] = mapped_column(String(80), nullable=False)
	status: Mapped[str] = mapped_column(String(30), nullable=False)
	delivered_item_sku: Mapped[str | None] = mapped_column(String(40))
	delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

	order: Mapped[Order] = relationship(back_populates="shipments")


class Payment(Base):
	"""Payment attempt recorded for an order."""

	__tablename__ = "payments"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	order_id: Mapped[str] = mapped_column(
		ForeignKey("orders.id"), nullable=False, index=True
	)
	amount: Mapped[float] = mapped_column(Float, nullable=False)
	status: Mapped[str] = mapped_column(String(20), nullable=False)
	gateway_ref: Mapped[str] = mapped_column(String(80), nullable=False)
	amount_deducted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)

	order: Mapped[Order] = relationship(back_populates="payments")


class Ticket(Base):
	"""Support ticket with investigation, actions, and escalation details."""

	__tablename__ = "tickets"
	__table_args__ = (
		CheckConstraint(
			"status IN ('opened', 'assigned', 'pending_customer', 'resolved', 'closed')",
			name="ck_tickets_status",
		),
	)

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	customer_id: Mapped[str] = mapped_column(
		ForeignKey("customers.id"), nullable=False, index=True
	)
	order_id: Mapped[str | None] = mapped_column(ForeignKey("orders.id"), index=True)
	intent: Mapped[str] = mapped_column(String(50), nullable=False)
	category: Mapped[str] = mapped_column(String(80), nullable=False)
	priority: Mapped[str] = mapped_column(String(20), nullable=False)
	sentiment: Mapped[str] = mapped_column(String(30), nullable=False)
	summary: Mapped[str] = mapped_column(Text, nullable=False)
	investigation: Mapped[str] = mapped_column(Text, nullable=False)
	actions_taken: Mapped[list[dict[str, Any]]] = mapped_column(
		JSON, nullable=False, default=list
	)
	status: Mapped[str] = mapped_column(String(30), nullable=False, default="opened")
	assigned_to: Mapped[str | None] = mapped_column(String(120))
	escalation_reason: Mapped[str | None] = mapped_column(Text)
	sla_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
	sla_risk: Mapped[bool] = mapped_column(
		Boolean, nullable=False, default=False, server_default="0"
	)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)
	updated_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
	)

	customer: Mapped[Customer] = relationship(back_populates="tickets")
	order: Mapped[Order | None] = relationship(back_populates="tickets")
	csat_responses: Mapped[list["CSAT"]] = relationship(back_populates="ticket")


class Conversation(Base):
	"""Conversation state persisted across customer messages."""

	__tablename__ = "conversations"

	id: Mapped[str] = mapped_column(String(80), primary_key=True)
	customer_id: Mapped[str] = mapped_column(
		ForeignKey("customers.id"), nullable=False, index=True
	)
	channel: Mapped[str] = mapped_column(String(30), nullable=False)
	state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
	started_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)

	customer: Mapped[Customer] = relationship(back_populates="conversations")
	messages: Mapped[list["Message"]] = relationship(
		back_populates="conversation", cascade="all, delete-orphan"
	)
	action_logs: Mapped[list["ActionLog"]] = relationship(
		back_populates="conversation"
	)
	csat_responses: Mapped[list["CSAT"]] = relationship(back_populates="conversation")


class Message(Base):
	"""A customer, agent, or system message in a conversation."""

	__tablename__ = "messages"
	__table_args__ = (
		CheckConstraint(
			"role IN ('customer', 'agent', 'system')", name="ck_messages_role"
		),
	)

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	conversation_id: Mapped[str] = mapped_column(
		ForeignKey("conversations.id"), nullable=False, index=True
	)
	role: Mapped[str] = mapped_column(String(20), nullable=False)
	content: Mapped[str] = mapped_column(Text, nullable=False)
	channel: Mapped[str] = mapped_column(String(30), nullable=False)
	timestamp: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)

	conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ActionLog(Base):
	"""Structured record of a deterministic support action."""

	__tablename__ = "action_log"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	conversation_id: Mapped[str | None] = mapped_column(
		ForeignKey("conversations.id"), index=True
	)
	tool_name: Mapped[str] = mapped_column(String(80), nullable=False)
	args: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	success: Mapped[bool] = mapped_column(Boolean, nullable=False)
	timestamp: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)

	conversation: Mapped[Conversation | None] = relationship(back_populates="action_logs")


class CSAT(Base):
	"""Customer satisfaction response associated with a ticket or conversation."""

	__tablename__ = "csat"
	__table_args__ = (
		CheckConstraint("rating BETWEEN 1 AND 5", name="ck_csat_rating"),
	)

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	ticket_id: Mapped[int | None] = mapped_column(ForeignKey("tickets.id"), index=True)
	conversation_id: Mapped[str | None] = mapped_column(
		ForeignKey("conversations.id"), index=True
	)
	rating: Mapped[int] = mapped_column(Integer, nullable=False)
	sentiment: Mapped[str | None] = mapped_column(String(30))
	resolution_quality: Mapped[str | None] = mapped_column(String(50))
	comment: Mapped[str | None] = mapped_column(Text)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), nullable=False, default=utc_now
	)

	ticket: Mapped[Ticket | None] = relationship(back_populates="csat_responses")
	conversation: Mapped[Conversation | None] = relationship(
		back_populates="csat_responses"
	)