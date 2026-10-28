"""Gemini-compatible input and output schemas for support agents."""

from typing import Literal

from pydantic import BaseModel, Field

IntentName = Literal[
	"wrong_item",
	"damaged_item",
	"missing_item",
	"order_status",
	"payment_failure",
	"refund_request",
	"cancel_order",
	"login_issue",
	"subscription_issue",
	"fraud_complaint",
	"legal_request",
	"data_breach",
	"general_query",
]
Priority = Literal["low", "medium", "high"]
Sentiment = Literal["positive", "neutral", "negative", "angry"]
EscalationChoice = Literal["auto_resolve", "escalate", "ask_more_info"]
TicketStatus = Literal[
	"opened", "assigned", "pending_customer", "resolved", "closed"
]


class ExtractedEntity(BaseModel):
	"""A named fact extracted from conversation text."""

	name: str
	value: str


class IntentResult(BaseModel):
	"""Validated intent, urgency, sentiment, and extracted entities."""

	intent: IntentName
	category: str
	priority: Priority
	sentiment: Sentiment
	extracted_entities: list[ExtractedEntity] = Field(default_factory=list)


class CustomerProfile(BaseModel):
	"""Customer profile fields read from the support database."""

	customer_id: str
	name: str
	email: str
	tier: str
	phone: str | None = None
	account_status: str
	failed_login_attempts: int
	created_at: str | None = None


class RecentOrder(BaseModel):
	"""Order facts used to provide customer context."""

	order_id: str
	status: str
	total: float
	payment_status: str
	placed_at: str
	delivered_at: str | None = None


class SubscriptionSummary(BaseModel):
	"""Subscription facts read from the customer profile."""

	status: str
	tier: str


class CustomerContext(BaseModel):
	"""Database-grounded customer, order, ticket, and subscription context."""

	profile: CustomerProfile
	tier: str
	recent_orders: list[RecentOrder] = Field(default_factory=list)
	past_tickets_count: int
	subscription: SubscriptionSummary
	summary: str


class ContextSummary(BaseModel):
	"""One-paragraph human-readable summary of database context."""

	summary: str


class KnowledgeArticle(BaseModel):
	"""A retrieved knowledge-base article or article chunk."""

	article_id: str
	title: str
	category: str
	reliability_score: float
	workflow_id: str
	content: str
	similarity: float = 0.0
	ranking_score: float = 0.0


class WorkflowSelection(BaseModel):
	"""LLM selection from the retrieved workflow candidates."""

	workflow_id: str
	confidence: float = Field(ge=0.0, le=1.0)
	reasoning: str


class KBResult(BaseModel):
	"""Retrieved knowledge and the selected deterministic workflow."""

	articles: list[KnowledgeArticle] = Field(default_factory=list)
	selected_workflow: str
	confidence: float = Field(ge=0.0, le=1.0)


class TroubleshootStep(BaseModel):
	"""One deterministic troubleshooting check and its factual finding."""

	step: str
	finding: str


class DiagnosisResponse(BaseModel):
	"""Customer-safe diagnosis phrasing generated from known facts."""

	diagnosis: str


class TroubleshootResult(BaseModel):
	"""Verified troubleshooting findings and the code-selected next action."""

	steps_run: list[TroubleshootStep] = Field(default_factory=list)
	diagnosis: str
	proposed_action: str


class ExecutedAction(BaseModel):
	"""A support action and its recorded outcome."""

	tool_name: str
	success: bool
	summary: str


class ActionResult(BaseModel):
	"""Outcome of one or more deterministic support actions."""

	actions_executed: list[ExecutedAction] = Field(default_factory=list)
	success: bool
	customer_message: str


class ConversationState(BaseModel):
	"""Persistable memory state for an in-progress conversation."""

	entities: list[ExtractedEntity] = Field(default_factory=list)
	missing_fields: list[str] = Field(default_factory=list)
	intent: str | None = None
	lifecycle_stage: Literal[
		"new", "gathering_info", "investigating", "acting", "resolved", "escalated"
	] = "new"
	turn_count: int = Field(ge=0)


class EscalationDecision(BaseModel):
	"""Validated decision about resolving, escalating, or clarifying an issue."""

	decision: EscalationChoice
	reason: str
	confidence: float = Field(ge=0.0, le=1.0)


class Ticket(BaseModel):
	"""Complete human-ready ticket matching the support workflow spec."""

	ticket_id: int | None = None
	customer_id: str
	customer_name: str
	customer_tier: str
	order_id: str | None = None
	intent: IntentName
	category: str
	priority: Priority
	sentiment: Sentiment
	summary: str
	investigation: str
	actions_taken: list[ExecutedAction] = Field(default_factory=list)
	status: TicketStatus = "opened"
	assigned_to: str | None = None
	escalation_reason: str | None = None
	sla_deadline: str | None = None
	conversation_summary: str
	suggested_next_step: str


class ActionMessage(BaseModel):
	"""Customer-facing wording derived solely from action tool results."""

	customer_message: str


class TicketText(BaseModel):
	"""LLM-authored issue summary and suggested human next step."""

	issue_summary: str
	suggested_next_step: str


class CSATResult(BaseModel):
	"""Structured customer-satisfaction score and labels."""

	rating: int = Field(ge=1, le=5)
	sentiment: Sentiment
	resolution_quality: Literal["resolved_fully", "partially", "not_resolved"]