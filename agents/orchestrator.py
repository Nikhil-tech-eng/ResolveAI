"""Coordinate support agents through LangGraph or an equivalent local state machine."""

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents import (
	action_agent,
	context_agent,
	csat_agent,
	escalation_agent,
	intent_agent,
	knowledge_agent,
	memory_agent,
	ticket_agent,
	tracking_agent,
	troubleshooting_agent,
)
from agents.schemas import (
	ActionResult,
	ConversationState,
	CustomerContext,
	EscalationDecision,
	ExtractedEntity,
	IntentResult,
	KBResult,
	Ticket,
	TroubleshootResult,
)
from db.models import Conversation, Customer, Message
from db.session import SessionLocal

logger = logging.getLogger(__name__)
ORDER_ID_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)
LANGGRAPH_AVAILABLE = False


class TraceEntry(TypedDict):
	"""One observable execution step in the customer support trace."""

	agent: str
	input_summary: str
	output_summary: str
	tools_called: list[str]
	duration_ms: int
	confidence: float


class AgentState(TypedDict, total=False):
	"""Shared LangGraph state for one customer support turn."""

	messages: list[dict[str, str]]
	conversation_id: str
	customer_id: str
	channel: str
	message: str
	intent: IntentResult
	context: CustomerContext
	kb: KBResult
	troubleshoot: TroubleshootResult
	actions: ActionResult
	escalation: EscalationDecision
	ticket: Ticket
	trace: list[TraceEntry]
	confidence: float
	memory: ConversationState
	entities: dict[str, str]
	proposed_action: str
	policy_blocked: bool
	customer_disputes: bool
	attempts: int
	reply: str
	awaiting_csat: bool
	csat_result: Any
	tracking_result: dict[str, Any]


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use the supplied database session or create a managed one."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _plain(value: Any) -> Any:
	"""Convert Pydantic values to plain data for compact summaries and state output."""
	if hasattr(value, "model_dump"):
		return value.model_dump(mode="json")
	return value


def _compact(value: Any, limit: int = 240) -> str:
	"""Render a value as a short trace summary."""
	text = str(_plain(value))
	return text if len(text) <= limit else text[: limit - 3] + "..."


def _trace_node(
	name: str,
	handler: Any,
	tools: list[str] | None = None,
) -> Any:
	"""Wrap one graph node so every execution emits the required trace record."""
	def run(state: AgentState) -> AgentState:
		started = time.perf_counter()
		input_summary = _compact(state.get("message", state.get("intent", "")))
		handler(state)
		output_value = state.get(
			"reply",
			state.get(
				"ticket",
				state.get(
					"escalation",
					state.get("actions", state.get("troubleshoot", state.get("intent", "done"))),
				),
			),
		)
		trace: TraceEntry = {
			"agent": name,
			"input_summary": input_summary,
			"output_summary": _compact(output_value),
			"tools_called": tools or [],
			"duration_ms": int((time.perf_counter() - started) * 1000),
			"confidence": float(state.get("confidence", 0.0) or 0.0),
		}
		state.setdefault("trace", []).append(trace)
		return state
	return run


def _is_dispute(message: str) -> bool:
	"""Detect an explicit customer dispute of a blocked policy action."""
	text = message.lower()
	return any(
		phrase in text
		for phrase in ("i disagree", "i dispute", "that's unfair", "this is unfair", "not acceptable", "why can't you")
	)


def _failed_troubleshooting(result: TroubleshootResult) -> int:
	"""Count troubleshooting findings that explicitly report failed checks."""
	markers = ("failed", "error", "not found", "could not", "blocked")
	return sum(
		1
		for step in result.steps_run
		if any(marker in step.finding.lower() for marker in markers)
	)


def _persist_failed_attempt(state: AgentState, session: Session) -> int:
	"""Increment the persisted failed-action count for a conversation."""
	conversation = session.get(Conversation, state["conversation_id"])
	if conversation is None:
		return 0
	conversation_state = dict(conversation.state or {})
	failed_attempts = int(conversation_state.get("failed_attempts", 0)) + 1
	conversation_state["failed_attempts"] = failed_attempts
	conversation.state = conversation_state
	session.commit()
	return failed_attempts

def _infer_order_id(state: AgentState, session: Session) -> None:
	"""Fill a missing order_id from the customer's most recent order."""
	memory = state.get("memory")
	if memory is None or "order_id" not in memory.missing_fields:
		return
	orders = getattr(state.get("context"), "recent_orders", None) or []
	if not orders:
		return
	first = orders[0]
	order_id = str(first["order_id"] if isinstance(first, dict) else first.order_id)
	memory.entities.append(ExtractedEntity(name="order_id", value=order_id))
	memory.missing_fields = memory_agent._derive_missing_fields(memory.intent, memory.entities)
	if not memory.missing_fields and memory.lifecycle_stage == "gathering_info":
		memory.lifecycle_stage = "investigating"
	state["entities"]["order_id"] = order_id
	memory_agent.save_state(state["conversation_id"], memory, session=session)

def _build_handlers(session: Session) -> dict[str, Any]:
	"""Create node handlers bound to the current request's database session."""
	def memory_load(state: AgentState) -> None:
		state["memory"] = memory_agent.load_state(state["conversation_id"], session=session)
		conversation = session.get(Conversation, state["conversation_id"])
		state["attempts"] = int((conversation.state or {}).get("failed_attempts", 0)) if conversation else 0

	def intent(state: AgentState) -> None:
		history = [f"{item['role']}: {item['content']}" for item in state["messages"]]
		result = intent_agent.detect_intent(history)
		if result.sentiment == "angry":
			result.priority = "high"
		state["intent"] = result
		memory = memory_agent.update_state(
			state["conversation_id"],
			result.intent,
			entities=result.extracted_entities,
			session=session,
		)
		state["memory"] = memory
		entities = {entity.name: entity.value for entity in memory.entities}
		entities.update({entity.name: entity.value for entity in result.extracted_entities})
		entities["customer_id"] = state["customer_id"]
		state["entities"] = entities
		state["policy_blocked"] = False
		state["customer_disputes"] = _is_dispute(state["message"])

	def context(state: AgentState) -> None:
		state["context"] = context_agent.build_context(
			state["customer_id"], session=session
		)
		_infer_order_id(state, session)

	def ask_customer(state: AgentState) -> None:
		question = memory_agent.next_question(state.get("memory", ConversationState(turn_count=0)))
		escalation = state.get("escalation")
		if question is None and escalation is not None:
			question = escalation.reason
		if question is None:
			question = "Could you share a little more detail so I can help?"
		if state.get("intent") and state["intent"].sentiment in {"negative", "angry"}:
			question = f"I'm sorry you're dealing with this. {question}"
		state["reply"] = question
		state["memory"].lifecycle_stage = "gathering_info"
		memory_agent.save_state(state["conversation_id"], state["memory"], session=session)

	def knowledge(state: AgentState) -> None:
		intent_name = state["intent"].intent
		query = f"{intent_name} {state['message']}"
		state["kb"] = knowledge_agent.select_workflow(
			state["intent"], state["context"], query
		)
		state["confidence"] = state["kb"].confidence

	def troubleshoot(state: AgentState) -> None:
		result = troubleshooting_agent.troubleshoot(
			state["intent"],
			state["entities"],
			state["kb"].selected_workflow,
			session=session,
		)
		state["troubleshoot"] = result
		state["proposed_action"] = result.proposed_action
		state["policy_blocked"] = result.proposed_action.startswith(
			("escalate_policy", "escalate_refund")
		)
		state["customer_disputes"] = _is_dispute(state["message"])

	def escalation_check(state: AgentState) -> None:
		memory = state.get("memory", ConversationState(turn_count=0))
		failed_steps = _failed_troubleshooting(
			state.get("troubleshoot", TroubleshootResult(steps_run=[], diagnosis="", proposed_action=""))
		)
		state["attempts"] = int(state.get("attempts", 0)) + failed_steps
		conversation = session.get(Conversation, state["conversation_id"])
		stored_state = conversation.state or {} if conversation else {}
		state["escalation"] = escalation_agent.decide(
			{
				"intent": state.get("intent"),
				"context": state.get("context"),
				"confidence": state.get("confidence", 0.0),
				"sentiment": state.get("intent").sentiment if state.get("intent") else "neutral",
				"attempts": state["attempts"],
				"unresolved": memory.lifecycle_stage != "resolved",
				"policy_blocked": state.get("policy_blocked", False),
				"customer_disputes": state.get("customer_disputes", False),
				"steps_run": [step.model_dump(mode="json") for step in state.get("troubleshoot").steps_run] if state.get("troubleshoot") else [],
				"sla_at_risk": bool(stored_state.get("sla_at_risk", False)),
				"tier": state.get("context").tier if state.get("context") else "standard",
			}
		)

	def action(state: AgentState) -> None:
		proposed_action = state.get("proposed_action", "")
		intent_name = state["intent"].intent
		if proposed_action == "schedule_replacement":
			state["entities"]["verified"] = "true"
		if proposed_action == "escalate_policy_block" and intent_name == "cancel_order":
			proposed_action = "cancel"
		if proposed_action in action_agent.ACTION_SEQUENCES:
			state["memory"].lifecycle_stage = "acting"
			memory_agent.save_state(state["conversation_id"], state["memory"], session=session)
			result = action_agent.execute(
				proposed_action,
				state.get("entities", {}),
				state["customer_id"],
				state["conversation_id"],
				session=session,
			)
		elif proposed_action == "share_verified_order_status":
			angry_unresolved = state["intent"].sentiment == "angry"
			result = ActionResult(
				actions_executed=[],
				success=not angry_unresolved,
				customer_message=state["troubleshoot"].diagnosis,
			)
		else:
			result = ActionResult(
				actions_executed=[],
				success=False,
				customer_message=state["troubleshoot"].diagnosis,
			)
		state["actions"] = result
		if result.success:
			state["memory"].lifecycle_stage = "resolved"
			memory_agent.save_state(state["conversation_id"], state["memory"], session=session)
		else:
			state["attempts"] = _persist_failed_attempt(state, session)

	def ticket(state: AgentState) -> None:
		if state["intent"].sentiment == "angry":
			state["intent"].priority = "high"
		state["memory"].lifecycle_stage = "escalated"
		memory_agent.save_state(state["conversation_id"], state["memory"], session=session)
		state["ticket"] = ticket_agent.generate_ticket(state, session=session)
		conversation = session.get(Conversation, state["conversation_id"])
		if conversation is not None:
			conversation.state = {**conversation.state, "ticket_id": state["ticket"].ticket_id}
			session.commit()

	def tracking(state: AgentState) -> None:
		ticket_result = state.get("ticket")
		if ticket_result is None or ticket_result.ticket_id is None:
			return
		state["tracking_result"] = tracking_agent.transition(
			ticket_result.ticket_id,
			"assigned",
			"support_orchestrator",
			session=session,
		)
		state["ticket"] = ticket_result.model_copy(update={"status": "assigned"})

	def reply(state: AgentState) -> None:
		intent_result = state.get("intent")
		sentiment = intent_result.sentiment if intent_result else "neutral"
		if state.get("escalation") and state["escalation"].decision == "escalate":
			ticket_id = state.get("ticket").ticket_id if state.get("ticket") else "pending"
			text = f"I've sent this to our support team for review. Your ticket number is {ticket_id}."
		elif state.get("actions") and not state["actions"].success:
			failed = next(
				(action for action in state["actions"].actions_executed if not action.success),
				None,
			)
			text = (
				f"I couldn't complete that request: {failed.summary}"
				if failed is not None
				else state["actions"].customer_message
			)
		elif state.get("actions"):
			text = state["actions"].customer_message
		else:
			text = state.get("troubleshoot").diagnosis if state.get("troubleshoot") else "I have recorded your request."
		if sentiment in {"negative", "angry"} and not text.lower().startswith("i'm sorry"):
			text = f"I'm sorry you're dealing with this. {text}"
		state["reply"] = text

	def csat_prompt(state: AgentState) -> None:
		actions = state.get("actions")
		if actions is None or not actions.success:
			return
		state["memory"].lifecycle_stage = "resolved"
		memory_agent.save_state(state["conversation_id"], state["memory"], session=session)
		conversation = session.get(Conversation, state["conversation_id"])
		if conversation is not None:
			conversation.state = {**conversation.state, "awaiting_csat": True}
			session.commit()
		state["awaiting_csat"] = True
		state["reply"] = (
			state.get("reply", "")
			+ "\n\nPlease rate your support experience from 1 to 5."
		)

	handlers = {
		"memory_load": memory_load,
		"intent": intent,
		"context": context,
		"ask_customer": ask_customer,
		"knowledge": knowledge,
		"troubleshoot": troubleshoot,
		"escalation_check": escalation_check,
		"action": action,
		"ticket": ticket,
		"tracking": tracking,
		"reply": reply,
		"csat_prompt": csat_prompt,
	}
	tool_names = {
		"memory_load": ["memory_agent.load_state"],
		"intent": ["intent_agent.detect_intent", "memory_agent.update_state"],
		"context": ["context_agent.build_context", "tools.read_tools"],
		"ask_customer": ["memory_agent.next_question"],
		"knowledge": ["kb.search.search_kb", "knowledge_agent.select_workflow"],
		"troubleshoot": ["troubleshooting_agent.troubleshoot", "tools.read_tools"],
		"escalation_check": ["escalation_agent.decide"],
		"action": ["action_agent.execute", "tools.action_tools"],
		"ticket": ["ticket_agent.generate_ticket"],
		"tracking": ["tracking_agent.transition", "tools.action_tools.send_notification"],
	}
	return {
		name: _trace_node(name, handler, tool_names.get(name, []))
		for name, handler in handlers.items()
	}


def _route_context(state: AgentState) -> str:
	"""Route to clarification when intent-specific required fields are missing."""
	return "ask_customer" if state.get("memory") and state["memory"].missing_fields else "knowledge"


def _route_escalation(state: AgentState) -> str:
	"""Route based on the escalation agent's validated decision."""
	decision = state.get("escalation")
	return decision.decision if decision else "escalate"


def _route_reply(state: AgentState) -> str:
	"""Only the normal auto-resolution path continues to the CSAT prompt."""
	if state.get("escalation") and state["escalation"].decision == "escalate":
		return "done"
	return "csat_prompt"


class _ManualStateGraph:
	"""Small StateGraph-compatible executor used when LangGraph is unavailable."""

	def __init__(self, nodes: dict[str, Any]) -> None:
		self.nodes = nodes

	def invoke(self, state: AgentState) -> AgentState:
		"""Execute nodes using the same conditional routes as the LangGraph graph."""
		node_name: str | None = "memory_load"
		while node_name is not None:
			self.nodes[node_name](state)
			if node_name == "memory_load":
				node_name = "intent"
			elif node_name == "intent":
				node_name = "context"
			elif node_name == "context":
				node_name = _route_context(state)
			elif node_name == "ask_customer":
				node_name = None
			elif node_name == "knowledge":
				node_name = "troubleshoot"
			elif node_name == "troubleshoot":
				node_name = "escalation_check"
			elif node_name == "escalation_check":
				route = _route_escalation(state)
				node_name = {
					"escalate": "ticket",
					"auto_resolve": "action",
					"ask_more_info": "ask_customer",
				}.get(route, "ticket")
			elif node_name == "action":
				node_name = "reply"
			elif node_name == "ticket":
				node_name = "tracking"
			elif node_name == "tracking":
				node_name = "reply"
			elif node_name == "reply":
				node_name = None if _route_reply(state) == "done" else "csat_prompt"
			else:
				node_name = None
		return state


class _ClosingGraph:
	"""Close a graph-owned session after invocation, not during graph compilation."""

	def __init__(self, graph: Any, session: Session) -> None:
		self.graph = graph
		self.session = session

	def invoke(self, state: AgentState) -> AgentState:
		"""Invoke the graph and close the session created by build_graph."""
		try:
			return self.graph.invoke(state)
		finally:
			self.session.close()


def build_graph(session: Session | None = None) -> Any:
	"""Build and compile the requested StateGraph, or its manual equivalent."""
	global LANGGRAPH_AVAILABLE
	owns_session = session is None
	if owns_session:
		session = SessionLocal()
	nodes = _build_handlers(session)
	try:
		from langgraph.graph import END, START, StateGraph

		LANGGRAPH_AVAILABLE = True
	except Exception as error:
		logger.warning("LangGraph unavailable; using manual state machine: %s", error)
		LANGGRAPH_AVAILABLE = False
		manual_graph = _ManualStateGraph(nodes)
		return _ClosingGraph(manual_graph, session) if owns_session else manual_graph
	try:
		graph = StateGraph(AgentState)
		for name, node in nodes.items():
			graph.add_node(name, node)
		graph.add_edge(START, "memory_load")
		graph.add_edge("memory_load", "intent")
		graph.add_edge("intent", "context")
		graph.add_conditional_edges(
			"context",
			_route_context,
			{"ask_customer": "ask_customer", "knowledge": "knowledge"},
		)
		graph.add_edge("ask_customer", END)
		graph.add_edge("knowledge", "troubleshoot")
		graph.add_edge("troubleshoot", "escalation_check")
		graph.add_conditional_edges(
			"escalation_check",
			_route_escalation,
			{
				"escalate": "ticket",
				"auto_resolve": "action",
				"ask_more_info": "ask_customer",
			},
		)
		graph.add_edge("action", "reply")
		graph.add_edge("ticket", "tracking")
		graph.add_edge("tracking", "reply")
		graph.add_conditional_edges(
			"reply",
			_route_reply,
			{"done": END, "csat_prompt": "csat_prompt"},
		)
		graph.add_edge("csat_prompt", END)
		compiled = graph.compile()
	except Exception:
		logger.exception("Could not compile LangGraph; using equivalent manual graph.")
		compiled = _ManualStateGraph(nodes)
	return _ClosingGraph(compiled, session) if owns_session else compiled


def _append_agent_message(session: Session, conversation_id: str, message: str, channel: str) -> None:
	"""Persist the final assistant message for one request."""
	session.add(
		Message(
			conversation_id=conversation_id,
			role="agent",
			content=message,
			channel=channel,
		)
	)
	session.commit()


def _followup_csat(
	conversation: Conversation,
	message: str,
	channel: str,
	session: Session,
) -> dict[str, Any]:
	"""Collect a rating entered after the graph's CSAT prompt."""
	started = time.perf_counter()
	rating_match = re.search(r"\b([1-5])\b", message)
	if rating_match is None:
		reply = "Please send a rating from 1 to 5 for your support experience."
		_append_agent_message(session, conversation.id, reply, channel)
		trace: TraceEntry = {
			"agent": "csat_prompt",
			"input_summary": _compact(message),
			"output_summary": "Requested a valid 1-5 rating.",
			"tools_called": [],
			"duration_ms": int((time.perf_counter() - started) * 1000),
			"confidence": 1.0,
		}
		return {"reply": reply, "trace": [trace], "state": {"awaiting_csat": True}, "ticket": None}

	rating = int(rating_match.group(1))
	comment = (message[: rating_match.start()] + message[rating_match.end() :]).strip(" ,.!\t\r\n")
	result = csat_agent.collect(
		conversation.id,
		rating,
		comment or None,
		session=session,
	)
	intent = memory_agent.load_state(conversation.id, session=session).intent or "general_query"
	memory = memory_agent.update_state(
		conversation.id,
		intent,
		lifecycle_stage="resolved",
		session=session,
	)
	conversation = session.get(Conversation, conversation.id)
	if conversation is not None:
		conversation.state = {
			**conversation.state,
			"awaiting_csat": False,
			"csat_result": result.model_dump(mode="json"),
		}
		session.commit()
	reply = "Thanks for sharing your feedback."
	_append_agent_message(session, conversation.id, reply, channel)
	trace = {
		"agent": "csat_prompt",
		"input_summary": _compact(message),
		"output_summary": _compact(result),
		"tools_called": ["csat_agent.collect"],
		"duration_ms": int((time.perf_counter() - started) * 1000),
		"confidence": 1.0,
	}
	return {
		"reply": reply,
		"trace": [trace],
		"state": {"conversation_id": conversation.id, "memory": memory, "csat": result},
		"ticket": None,
	}


def handle_message(
	conversation_id: str,
	customer_id: str,
	message: str,
	channel: str,
	session: Session | None = None,
) -> dict[str, Any]:
	"""Persist the customer turn, run the agent graph, and persist its reply."""
	with _using_session(session) as db:
		if db.get(Customer, customer_id) is None:
			raise ValueError(f"Customer {customer_id} was not found.")
		conversation = db.get(Conversation, conversation_id)
		if conversation is None:
			conversation = Conversation(
				id=conversation_id,
				customer_id=customer_id,
				channel=channel,
				state={},
				status="open",
			)
			db.add(conversation)
			db.commit()
		elif conversation.customer_id != customer_id:
			raise ValueError("Conversation does not belong to the specified customer.")

		db.add(
			Message(
				conversation_id=conversation_id,
				role="customer",
				content=message,
				channel=channel,
				timestamp=datetime.now(timezone.utc),
			)
		)
		db.commit()
		conversation = db.get(Conversation, conversation_id)
		if conversation and conversation.state.get("awaiting_csat"):
			return _followup_csat(conversation, message, channel, db)

		stored_messages = db.scalars(
			select(Message)
			.where(Message.conversation_id == conversation_id)
			.order_by(Message.timestamp, Message.id)
		).all()
		state: AgentState = {
			"conversation_id": conversation_id,
			"customer_id": customer_id,
			"channel": channel,
			"message": message,
			"messages": [
				{"role": item.role, "content": item.content}
				for item in stored_messages
			],
			"trace": [],
			"confidence": 1.0,
		}
		result = build_graph(db).invoke(state)
		reply_text = result.get("reply", "I have recorded your request and will help with the next step.")
		_append_agent_message(db, conversation_id, reply_text, channel)
		return {
			"reply": reply_text,
			"trace": result.get("trace", []),
			"state": result,
			"ticket": result.get("ticket"),
		}