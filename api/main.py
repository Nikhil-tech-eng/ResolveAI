"""ResolveAI FastAPI application and database-backed support endpoints."""

import os
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from agents import csat_agent, kb_expansion, orchestrator, tracking_agent
from agents.schemas import CSATResult, Ticket as AgentTicket, TicketStatus
from agents.kb_expansion import KBReviewResult, PendingKBArticle
from analytics.root_cause import RootCauseAlert, detect_root_cause_spikes
from db.models import CSAT, Conversation, Customer, Message, Ticket
from db.session import SessionLocal, init_db

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
	"""Customer message sent to the ResolveAI orchestrator."""

	conversation_id: str | None = None
	customer_id: str
	message: str = Field(min_length=1)
	channel: str = "web"


class TraceRecord(BaseModel):
	"""One agent execution item returned by the orchestrator."""

	agent: str
	input_summary: str
	output_summary: str
	tools_called: list[str]
	duration_ms: int
	confidence: float


class TicketResponse(BaseModel):
	"""Ticket fields exposed to the chat and supervisor UI."""

	id: int
	customer_id: str
	customer_name: str
	customer_tier: str
	order_id: str | None = None
	intent: str
	category: str
	priority: str
	sentiment: str
	summary: str
	investigation: str
	actions_taken: list[dict[str, Any]]
	status: TicketStatus
	assigned_to: str | None = None
	escalation_reason: str | None = None
	sla_deadline: datetime | None = None
	sla_risk: bool
	created_at: datetime
	updated_at: datetime


class ChatResponse(BaseModel):
	"""Reply, trace, final graph state, and optional escalated ticket."""

	conversation_id: str
	reply: str
	trace: list[TraceRecord]
	state: dict[str, Any]
	ticket: AgentTicket | None = None


class TicketStatusRequest(BaseModel):
	"""Requested ticket lifecycle transition."""

	status: TicketStatus
	actor: str = "api"


class CSATRequest(BaseModel):
	"""Customer feedback submitted for a conversation."""

	conversation_id: str
	rating: int = Field(ge=1, le=5)
	comment: str | None = None


class CustomerOption(BaseModel):
	"""Customer identity option used by the chat UI."""

	id: str
	name: str
	tier: str


class IntentMetric(BaseModel):
	"""Ticket count grouped by intent."""

	intent: str
	count: int


class TimeMetric(BaseModel):
	"""Daily ticket or escalation count."""

	date: str
	count: int


class MetricsResponse(BaseModel):
	"""Supervisor dashboard KPI aggregates."""

	open_tickets: int
	resolution_rate: float
	avg_resolution_time_minutes: float | None
	escalation_rate: float
	avg_csat: float | None
	intent_distribution: list[IntentMetric]
	tickets_over_time: list[TimeMetric]
	escalations_over_time: list[TimeMetric]


class OutageSimulationResponse(BaseModel):
	"""Number of simulated payment-failure conversations created."""

	intent: str
	count: int
	window_minutes: int


class KBReviewRequest(BaseModel):
	"""Human decision for an auto-generated article."""

	decision: Literal["approve", "reject"]


def _sla_monitor(stop_event: threading.Event) -> None:
	"""Run the deterministic ticket SLA checker every 60 seconds."""
	while not stop_event.wait(60):
		try:
			tracking_agent.sla_check()
		except Exception:
			logger.exception("Background SLA check failed")


@asynccontextmanager
async def lifespan(application: FastAPI):
	"""Initialize the DB and manage the background SLA monitoring thread."""
	init_db()
	with SessionLocal() as session:
		tracking_agent.sla_check(session)
	stop_event = threading.Event()
	monitor = threading.Thread(
		target=_sla_monitor,
		args=(stop_event,),
		name="resolveai-sla-monitor",
		daemon=True,
	)
	application.state.sla_stop_event = stop_event
	monitor.start()
	try:
		yield
	finally:
		stop_event.set()
		monitor.join(timeout=5)


app = FastAPI(title="ResolveAI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
	allow_credentials=False,
	allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
	allow_headers=["*"],
)


def _ticket_response(ticket: Ticket, customer: Customer) -> TicketResponse:
	"""Convert a database ticket and its customer into the public schema."""
	return TicketResponse(
		id=ticket.id,
		customer_id=ticket.customer_id,
		customer_name=customer.name,
		customer_tier=customer.tier,
		order_id=ticket.order_id,
		intent=ticket.intent,
		category=ticket.category,
		priority=ticket.priority,
		sentiment=ticket.sentiment,
		summary=ticket.summary,
		investigation=ticket.investigation,
		actions_taken=ticket.actions_taken,
		status=ticket.status,
		assigned_to=ticket.assigned_to,
		escalation_reason=ticket.escalation_reason,
		sla_deadline=ticket.sla_deadline,
		sla_risk=ticket.sla_risk,
		created_at=ticket.created_at,
		updated_at=ticket.updated_at,
	)


@app.get("/customers", response_model=list[CustomerOption])
def list_customers() -> list[CustomerOption]:
	"""Return customer names and tiers for the chat selector."""
	with SessionLocal() as session:
		customers = session.scalars(select(Customer).order_by(Customer.id)).all()
		return [CustomerOption(id=item.id, name=item.name, tier=item.tier) for item in customers]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
	"""Pass a customer message through the ResolveAI agent graph."""
	conversation_id = request.conversation_id or f"conv-{uuid4()}"
	try:
		result = orchestrator.handle_message(
			conversation_id=conversation_id,
			customer_id=request.customer_id,
			message=request.message,
			channel=request.channel,
		)
		return ChatResponse(
			conversation_id=conversation_id,
			reply=result["reply"],
			trace=result["trace"],
			state=result["state"],
			ticket=result.get("ticket"),
		)
	except ValueError as error:
			status_code = 404 if "not found" in str(error).lower() else 400
			raise HTTPException(status_code=status_code, detail=str(error)) from error


@app.get("/tickets", response_model=list[TicketResponse])
def list_tickets(
	status: TicketStatus | None = Query(default=None),
) -> list[TicketResponse]:
	"""List tickets, optionally filtered by lifecycle status."""
	with SessionLocal() as session:
		statement = select(Ticket).order_by(Ticket.created_at.desc(), Ticket.id.desc())
		if status is not None:
			statement = statement.where(Ticket.status == status)
		tickets = session.scalars(statement).all()
		return [_ticket_response(ticket, ticket.customer) for ticket in tickets]


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int) -> TicketResponse:
	"""Return one ticket with its customer details."""
	with SessionLocal() as session:
		ticket = session.get(Ticket, ticket_id)
		if ticket is None:
			raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} was not found.")
		return _ticket_response(ticket, ticket.customer)


@app.patch("/tickets/{ticket_id}", response_model=TicketResponse)
def update_ticket(ticket_id: int, request: TicketStatusRequest) -> TicketResponse:
	"""Apply a valid ticket state transition and return the updated ticket."""
	with SessionLocal() as session:
		try:
			tracking_agent.transition(ticket_id, request.status, request.actor, session=session)
		except ValueError as error:
			status_code = 404 if "not found" in str(error).lower() else 409
			raise HTTPException(status_code=status_code, detail=str(error)) from error
		if request.status == "resolved":
			try:
				kb_expansion.draft_faq(ticket_id, session=session)
			except Exception:
				logger.exception("Resolved ticket %s could not produce an FAQ draft", ticket_id)
		ticket = session.get(Ticket, ticket_id)
		if ticket is None:
			raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} was not found.")
		return _ticket_response(ticket, ticket.customer)


@app.post("/csat", response_model=CSATResult)
def submit_csat(request: CSATRequest) -> CSATResult:
	"""Persist a conversation rating and clear its pending feedback flag."""
	with SessionLocal() as session:
		try:
			result = csat_agent.collect(
				request.conversation_id,
				request.rating,
				request.comment,
				session=session,
			)
		except ValueError as error:
			status_code = 404 if "not found" in str(error).lower() else 400
			raise HTTPException(status_code=status_code, detail=str(error)) from error
		conversation = session.get(Conversation, request.conversation_id)
		if conversation is not None:
			conversation.state = {**conversation.state, "awaiting_csat": False}
			session.commit()
		return result


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
	"""Compute live ticket and CSAT metrics with SQLite aggregate queries."""
	with SessionLocal() as session:
		total_tickets = int(session.scalar(select(func.count(Ticket.id))) or 0)
		open_tickets = int(
			session.scalar(
				select(func.count(Ticket.id)).where(
					Ticket.status.in_(("opened", "assigned", "pending_customer"))
				)
			)
			or 0
		)
		resolved_tickets = int(
			session.scalar(
				select(func.count(Ticket.id)).where(Ticket.status.in_(("resolved", "closed")))
			)
			or 0
		)
		resolution_rate = resolved_tickets / total_tickets if total_tickets else 0.0
		average_minutes = session.scalar(
			select(
				func.avg(
					(func.julianday(Ticket.updated_at) - func.julianday(Ticket.created_at))
					* 24
					* 60
				)
			).where(Ticket.status.in_(("resolved", "closed")))
		)
		escalated_tickets = int(
			session.scalar(
				select(func.count(Ticket.id)).where(
					or_(Ticket.escalation_reason.is_not(None), Ticket.assigned_to == "support_queue")
				)
			)
			or 0
		)
		average_csat = session.scalar(select(func.avg(CSAT.rating)))
		intent_rows = session.execute(
			select(Ticket.intent, func.count(Ticket.id))
			.group_by(Ticket.intent)
			.order_by(func.count(Ticket.id).desc())
		).all()
		day = func.date(Ticket.created_at)
		ticket_days = session.execute(
			select(day, func.count(Ticket.id)).group_by(day).order_by(day)
		).all()
		escalation_days = session.execute(
			select(day, func.count(Ticket.id))
			.where(Ticket.escalation_reason.is_not(None))
			.group_by(day)
			.order_by(day)
		).all()
		return MetricsResponse(
			open_tickets=open_tickets,
			resolution_rate=resolution_rate,
			avg_resolution_time_minutes=(
				round(float(average_minutes), 2) if average_minutes is not None else None
			),
			escalation_rate=(escalated_tickets / total_tickets if total_tickets else 0.0),
			avg_csat=round(float(average_csat), 2) if average_csat is not None else None,
			intent_distribution=[IntentMetric(intent=str(intent), count=int(count)) for intent, count in intent_rows],
			tickets_over_time=[TimeMetric(date=str(date), count=int(count)) for date, count in ticket_days],
			escalations_over_time=[TimeMetric(date=str(date), count=int(count)) for date, count in escalation_days],
		)


@app.get("/alerts", response_model=list[RootCauseAlert])
def alerts() -> list[RootCauseAlert]:
	"""Return intent spikes detected in a rolling 30-minute window."""
	return detect_root_cause_spikes()


@app.post("/alerts/simulate-outage", response_model=OutageSimulationResponse)
def simulate_outage() -> OutageSimulationResponse:
	"""Inject 20 clustered payment-failure conversations for dashboard demos."""
	with SessionLocal() as session:
		customers = session.scalars(select(Customer).order_by(Customer.id)).all()
		if not customers:
			raise HTTPException(status_code=503, detail="No customers are available for the simulation.")
		now = datetime.now(timezone.utc)
		for index in range(20):
			customer = customers[index % len(customers)]
			conversation_id = f"SIM-PAYFAIL-{uuid4()}"
			started_at = now - timedelta(seconds=index * 20)
			session.add(
				Conversation(
					id=conversation_id,
					customer_id=customer.id,
					channel="dashboard_simulation",
					state={"intent": "payment_failure", "source": "simulated_outage"},
					status="open",
					started_at=started_at,
				)
			)
			session.add(
				Message(
					conversation_id=conversation_id,
					role="customer",
					content="Checkout says payment failed, but my bank shows a deduction.",
					channel="dashboard_simulation",
					timestamp=started_at,
				)
			)
		session.commit()
	return OutageSimulationResponse(
		intent="payment_failure",
		count=20,
		window_minutes=30,
	)


@app.get("/kb/pending", response_model=list[PendingKBArticle])
def pending_kb_articles() -> list[PendingKBArticle]:
	"""List auto-generated KB articles awaiting supervisor review."""
	return kb_expansion.list_pending_articles()


@app.post("/kb/articles/{article_id}/review", response_model=KBReviewResult)
def review_kb_article(article_id: str, request: KBReviewRequest) -> KBReviewResult:
	"""Approve or reject a pending article and refresh its vector index."""
	try:
		return kb_expansion.review_article(article_id, request.decision)
	except ValueError as error:
		status_code = 404 if "not found" in str(error).lower() else 409
		raise HTTPException(status_code=status_code, detail=str(error)) from error