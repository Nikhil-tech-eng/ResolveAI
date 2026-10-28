"""Load and persist compact per-conversation memory in the conversations table."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from agents.schemas import ConversationState, ExtractedEntity
from db.models import Conversation
from db.session import SessionLocal

MISSING_FIELDS_BY_INTENT: dict[str, tuple[str, ...]] = {
	"wrong_item": ("order_id",),
	"damaged_item": ("order_id", "item_photo"),
	"missing_item": ("order_id",),
	"order_status": ("order_id",),
	"payment_failure": ("order_id",),
	"refund_request": ("order_id",),
	"cancel_order": ("order_id",),
	"login_issue": (),
	"subscription_issue": (),
	"fraud_complaint": (),
	"legal_request": (),
	"data_breach": (),
	"general_query": (),
}

NEXT_QUESTIONS: dict[str, str] = {
	"order_id": "What is your order ID? It usually starts with ORD.",
	"item_photo": "Could you share a clear photo of the damaged item?",
}


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a supplied SQLAlchemy session or open a managed session."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _normalize_entities(
	entities: Mapping[str, Any] | list[ExtractedEntity] | None,
) -> list[ExtractedEntity]:
	"""Convert mappings or entity lists to unique named memory records."""
	if entities is None:
		return []
	if isinstance(entities, Mapping):
		return [
			ExtractedEntity(name=str(name), value=str(value))
			for name, value in entities.items()
		]
	return [
		entity if isinstance(entity, ExtractedEntity) else ExtractedEntity.model_validate(entity)
		for entity in entities
	]


def _derive_missing_fields(intent: str | None, entities: list[ExtractedEntity]) -> list[str]:
	"""Return required fields for this intent that are not in memory."""
	if intent is None:
		return []
	known_names = {entity.name for entity in entities}
	return [
		field
		for field in MISSING_FIELDS_BY_INTENT.get(intent, ())
		if field not in known_names
	]


def load_state(
	conversation_id: str,
	intent: str | None = None,
	session: Session | None = None,
) -> ConversationState:
	"""Load memory state from the conversation JSON, preserving unrelated state."""
	with _using_session(session) as db:
		conversation = db.get(Conversation, conversation_id)
		if conversation is None:
			return ConversationState(
				intent=intent,
				missing_fields=_derive_missing_fields(intent, []),
				lifecycle_stage="new",
				turn_count=0,
			)
		stored = conversation.state.get("memory_state", {})
		if not stored:
			legacy_intent = intent or conversation.state.get("intent")
			reserved_fields = {"intent", "scenario", "source", "sentiment"}
			legacy_entities = [
				ExtractedEntity(name=str(name), value=str(value))
				for name, value in conversation.state.items()
				if name not in reserved_fields and isinstance(value, (str, int, float, bool))
			]
			missing_fields = _derive_missing_fields(legacy_intent, legacy_entities)
			return ConversationState(
				entities=legacy_entities,
				intent=legacy_intent,
				missing_fields=missing_fields,
				lifecycle_stage="gathering_info" if missing_fields else "new",
				turn_count=0,
			)
		loaded = ConversationState.model_validate(stored)
		if intent is not None and loaded.intent != intent:
			loaded.intent = intent
			loaded.missing_fields = _derive_missing_fields(intent, loaded.entities)
		return loaded


def save_state(
	conversation_id: str,
	state: ConversationState,
	session: Session | None = None,
) -> ConversationState:
	"""Persist memory under the conversation's memory_state JSON key."""
	with _using_session(session) as db:
		conversation = db.get(Conversation, conversation_id)
		if conversation is None:
			raise ValueError(f"Conversation {conversation_id} was not found.")
		conversation.state = {
			**conversation.state,
			"memory_state": state.model_dump(mode="json"),
		}
		db.commit()
		return state


def update_state(
	conversation_id: str,
	intent: str,
	entities: Mapping[str, Any] | list[ExtractedEntity] | None = None,
	lifecycle_stage: str | None = None,
	session: Session | None = None,
) -> ConversationState:
	"""Merge one message's facts, recompute missing fields, and increment the turn."""
	state = load_state(conversation_id, intent=intent, session=session)
	merged_entities = {entity.name: entity for entity in state.entities}
	for entity in _normalize_entities(entities):
		merged_entities[entity.name] = entity
	state.entities = list(merged_entities.values())
	state.missing_fields = _derive_missing_fields(intent, state.entities)
	state.intent = intent
	state.turn_count += 1
	if lifecycle_stage is not None:
		state.lifecycle_stage = lifecycle_stage
	elif state.missing_fields:
		state.lifecycle_stage = "gathering_info"
	elif state.lifecycle_stage in {"new", "gathering_info"}:
		state.lifecycle_stage = "investigating"
	return save_state(conversation_id, state, session=session)


def next_question(state: ConversationState) -> str | None:
	"""Return the next intent-specific clarification question, if one is needed."""
	if not state.missing_fields:
		return None
	return NEXT_QUESTIONS.get(
		state.missing_fields[0],
		f"Could you provide {state.missing_fields[0].replace('_', ' ')}?",
	)