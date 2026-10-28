"""Intent classification with an LLM and deterministic safety fallback."""

import logging
import re

from agents.schemas import ExtractedEntity, IntentName, IntentResult
from llm.client import call_llm_json

INTENT_SYSTEM_PROMPT = """Classify the customer's current support issue.
Choose exactly one intent from: wrong_item, damaged_item, missing_item,
order_status, payment_failure, refund_request, cancel_order, login_issue,
subscription_issue, fraud_complaint, legal_request, data_breach, general_query.
Use category labels such as order, payment, account, subscription, safety, or general.
Priority rules: payment_failure, fraud_complaint, and a locked account are high.
Use sentiment positive, neutral, negative, or angry. Understand typos and Hinglish
(Hindi-English mix); do not require grammatically correct English. Extract IDs and
other explicit facts as named entities. Do not invent customer or order facts."""

logger = logging.getLogger(__name__)
ORDER_ID_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)

_INTENT_RULES: tuple[tuple[IntentName, tuple[str, ...]], ...] = (
	("fraud_complaint", ("fraud", "unauthorized", "unrecognized charge", "unknown charge", "not my purchase", "chori")),
	("legal_request", ("lawyer", "legal", "sue", "court", "subpoena")),
	("data_breach", ("data breach", "privacy breach", "leaked my data", "personal data leak")),
	("wrong_item", ("wrong item", "wrong product", "incorrect item", "different item", "galat item", "galat product", "not what i ordered")),
	("damaged_item", ("damaged", "broken", "cracked", "defective", "toota", "damage hua")),
	("missing_item", ("missing item", "item missing", "not in the package", "missing parcel", "nahi mila")),
	("payment_failure", ("payment failed", "payment failure", "failed payment", "amount deducted", "money deducted", "charged but", "paisa kata", "paise kat", "payment nahi hua")),
	("refund_request", ("refund", "return my", "return this", "money back", "paise wapas", "refund chahiye")),
	("cancel_order", ("cancel", "cencel", "cancell", "stop my order", "order rok")),
	("subscription_issue", ("subscription", "membership", "plan not active", "paid but not active")),
	("login_issue", ("locked", "lockout", "can't log in", "cannot log in", "login", "log in", "sign in", "password", "account band")),
	("order_status", ("where is my order", "track my order", "tracking", "delivery status", "order status", "shipped yet")),
)


def _history_text(history: str | list[str]) -> str:
	"""Normalize a string or list of messages into conversation text."""
	if isinstance(history, str):
		return history
	return "\n".join(str(message) for message in history)


def _category_for(intent: IntentName) -> str:
	"""Return a stable high-level category for an intent."""
	if intent in {"payment_failure", "refund_request"}:
		return "payment"
	if intent in {"wrong_item", "damaged_item", "missing_item", "order_status", "cancel_order"}:
		return "order"
	if intent in {"login_issue"}:
		return "account"
	if intent == "subscription_issue":
		return "subscription"
	if intent in {"fraud_complaint", "legal_request", "data_breach"}:
		return "safety"
	return "general"


def _fallback_intent(text: str) -> IntentResult:
	"""Classify with deterministic keywords when Gemini cannot provide a result."""
	normalized = text.lower()
	intent: IntentName = "general_query"
	for candidate, keywords in _INTENT_RULES:
		if any(keyword in normalized for keyword in keywords):
			intent = candidate
			break

	if intent in {"payment_failure", "fraud_complaint"} or (
		intent == "login_issue" and any(word in normalized for word in ("locked", "lockout", "account band"))
	):
		priority = "high"
	elif intent == "general_query":
		priority = "low"
	else:
		priority = "medium"

	positive_words = ("thank", "thanks", "great", "happy", "shukriya", "dhanyavaad")
	angry_words = ("furious", "angry", "ridiculous", "unacceptable", "gussa", "bekar")
	negative_words = ("disappointed", "upset", "problem", "issue", "frustrated", "pareshan")
	if any(word in normalized for word in angry_words):
		sentiment = "angry"
	elif any(word in normalized for word in negative_words):
		sentiment = "negative"
	elif any(word in normalized for word in positive_words):
		sentiment = "positive"
	else:
		sentiment = "neutral"

	entities = [
		ExtractedEntity(name="order_id", value=match.upper())
		for match in ORDER_ID_PATTERN.findall(text)
	]
	return IntentResult(
		intent=intent,
		category=_category_for(intent),
		priority=priority,
		sentiment=sentiment,
		extracted_entities=entities,
	)


def detect_intent(history: str | list[str]) -> IntentResult:
	"""Classify conversation history and recover explicit order IDs in code."""
	text = _history_text(history)
	try:
		result = call_llm_json(INTENT_SYSTEM_PROMPT, text, IntentResult, "intent")
		if not isinstance(result, IntentResult):
			result = IntentResult.model_validate(result)
		order_matches = {match.upper() for match in ORDER_ID_PATTERN.findall(text)}
		known_orders = {
			entity.value.upper()
			for entity in result.extracted_entities
			if entity.name == "order_id"
		}
		for order_id in sorted(order_matches - known_orders):
			result.extracted_entities.append(
				ExtractedEntity(name="order_id", value=order_id)
			)
		if result.intent in {"payment_failure", "refund_request", "fraud_complaint"} or (
			result.intent == "login_issue"
			and any(word in text.lower() for word in ("locked", "lockout", "account band"))
		):
			result.priority = "high"
		return result
	except Exception as error:
		logger.warning("Intent model failed; using keyword classification: %s", error)
		return _fallback_intent(text)