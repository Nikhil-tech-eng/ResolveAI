"""Run scripted Agentic ResolveAI and stateless RAG evaluations."""

import argparse
import csv
import json
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agents import (
	action_agent,
	context_agent,
	csat_agent,
	escalation_agent,
	intent_agent,
	knowledge_agent,
	ticket_agent,
	troubleshooting_agent,
)
from agents.orchestrator import handle_message
from agents.schemas import (
	ActionMessage,
	ContextSummary,
	DiagnosisResponse,
	EscalationDecision,
	IntentResult,
	TicketText,
	WorkflowSelection,
)
from data.seed_data import seed_database
from evals import baseline_rag

SCENARIOS_PATH = Path(__file__).with_name("test_conversations.json")
RESULTS_CSV_PATH = Path(__file__).with_name("results.csv")
RESULTS_MARKDOWN_PATH = Path(__file__).with_name("results.md")
RESULTS_CHART_PATH = Path(__file__).with_name("comparison.png")
ORDER_ID_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)

WORKFLOW_BY_INTENT = {
	"wrong_item": "wrong_item",
	"damaged_item": "damaged_item",
	"missing_item": "missing_item",
	"order_status": "order_status",
	"payment_failure": "payment_deducted",
	"refund_request": "refund_request",
	"cancel_order": "cancel_order",
	"login_issue": "account_locked",
	"subscription_issue": "subscription_sync",
	"fraud_complaint": "fraud_escalate",
	"legal_request": "legal_escalate",
	"data_breach": "legal_escalate",
	"general_query": "general_info",
}


class EvalConversation(BaseModel):
	"""One scripted customer conversation and its expected outcome."""

	customer_id: str
	customer_turns: list[str]
	expected_intent: str
	expected_decision: Literal["auto_resolve", "escalate"]
	expected_final_action: str
	policy_blocked: bool = False


class MockProvider:
	"""Return deterministic schema-valid LLM values for offline evaluation."""

	def __init__(self) -> None:
		self.case: EvalConversation | None = None
		self.turn_index = 0

	def respond(
		self, _system: str, user: str, schema: type[BaseModel], agent: str = "default"
	) -> BaseModel:
		"""Build a deterministic response for the active script and schema."""
		if self.case is None:
			raise RuntimeError("Mock evaluation has no active conversation.")
		if schema is IntentResult:
			category = {
				"wrong_item": "order",
				"damaged_item": "order",
				"missing_item": "order",
				"order_status": "order",
				"payment_failure": "payment",
				"refund_request": "payment",
				"cancel_order": "order",
				"login_issue": "account",
				"subscription_issue": "subscription",
				"fraud_complaint": "safety",
				"legal_request": "safety",
				"data_breach": "safety",
				"general_query": "general",
			}[self.case.expected_intent]
			lower_user = user.lower()
			sentiment = (
				"angry"
				if any(word in lower_user for word in ("angry", "furious", "gussa"))
				else "negative"
				if any(word in lower_user for word in ("frustrated", "upset", "disappointed"))
				else "neutral"
			)
			entities = [
				{"name": "order_id", "value": match.upper()}
				for match in ORDER_ID_PATTERN.findall(user)
			]
			if "verified" in lower_user or "photos" in lower_user:
				entities.append({"name": "verified", "value": "true"})
			priority = "high" if self.case.expected_intent in {"payment_failure", "fraud_complaint"} else "medium"
			return IntentResult(
				intent=self.case.expected_intent,
				category=category,
				priority=priority,
				sentiment=sentiment,
				extracted_entities=entities,
			)
		if schema is ContextSummary:
			return ContextSummary(summary="Mocked seeded customer context.")
		if schema is WorkflowSelection:
			request = json.loads(user)
			workflow = request["candidates"][0]["workflow_id"]
			return WorkflowSelection(
				workflow_id=workflow,
				confidence=0.95,
				reasoning="Deterministic mock selected the retrieved candidate.",
			)
		if schema is DiagnosisResponse:
			return DiagnosisResponse(diagnosis="I checked the available support records.")
		if schema is EscalationDecision:
			decision = self.case.expected_decision
			if (
				self.case.expected_intent == "order_status"
				and self.case.expected_decision == "escalate"
				and self.turn_index == 1
			):
				decision = "auto_resolve"
			return EscalationDecision(
				decision=decision,
				reason="Mocked evaluation decision.",
				confidence=0.95,
			)
		if schema is ActionMessage:
			tool_results = json.loads(user)
			failure = next(
				(
					entry["result"].get("error")
					for entry in tool_results
					if not entry["result"]["success"]
				),
				None,
			)
			message = f"Action was blocked: {failure}" if failure else "The requested support action completed."
			return ActionMessage(customer_message=message)
		if schema is TicketText:
			return TicketText(
				issue_summary=f"Customer reports {self.case.expected_intent.replace('_', ' ')}.",
				suggested_next_step="Review the investigation and contact the customer.",
			)
		if schema.__name__ == "CommentSentiment":
			from agents.csat_agent import CommentSentiment

			return CommentSentiment(sentiment="neutral")
		if schema.__name__ == "BaselineAnswer":
			return baseline_rag.BaselineAnswer(
				answer="Based on the retrieved article, here are the relevant support steps."
			)
		raise ValueError(f"No mock response is defined for {schema.__name__}.")


def _mock_search(
	provider: MockProvider,
	query: str,
	_category: str | None = None,
	_k: int = 4,
) -> list[dict[str, Any]]:
	"""Return a deterministic candidate without embeddings or network access."""
	if provider.case is None:
		return []
	workflow_id = WORKFLOW_BY_INTENT[provider.case.expected_intent]
	return [
		{
			"article_id": f"mock-{workflow_id}",
			"title": f"Mock {workflow_id.replace('_', ' ')} guide",
			"category": "evaluation",
			"content": f"Evaluation reference for query: {query}",
			"reliability_score": 0.9,
			"workflow_id": workflow_id,
			"similarity": 0.95,
			"ranking_score": 0.855,
		}
	]


def _install_mock_provider(provider: MockProvider) -> None:
	"""Patch every module-level LLM and retrieval boundary for offline runs."""
	modules = (
		intent_agent,
		context_agent,
		knowledge_agent,
		troubleshooting_agent,
		escalation_agent,
		action_agent,
		ticket_agent,
		csat_agent,
		baseline_rag,
	)
	for module in modules:
		module.call_llm_json = provider.respond
	knowledge_agent.search_kb = lambda query, category=None, k=4: _mock_search(
		provider, query, category, k
	)
	baseline_rag.search_kb = lambda query, category=None, k=4: _mock_search(
		provider, query, category, k
	)


def _install_delay(delay_seconds: float) -> None:
	"""Wrap direct LLM call sites with a delay for live Gemini evaluations."""
	if delay_seconds <= 0:
		return
	modules = (
		intent_agent,
		context_agent,
		knowledge_agent,
		troubleshooting_agent,
		escalation_agent,
		action_agent,
		ticket_agent,
		csat_agent,
		baseline_rag,
	)
	for module in modules:
		original_call = module.call_llm_json

		def delayed_call(*args: Any, _original: Callable[..., Any] = original_call, **kwargs: Any) -> Any:
			time.sleep(delay_seconds)
			return _original(*args, **kwargs)

		module.call_llm_json = delayed_call


def _actual_action(state: dict[str, Any]) -> str:
	"""Extract the final action or escalation label from one graph result."""
	escalation = state.get("escalation")
	if escalation is not None and escalation.decision == "escalate":
		return "escalate"
	actions = state.get("actions")
	if actions is not None:
		for action in actions.actions_executed:
			if action.tool_name != "send_notification":
				return action.tool_name
	return str(state.get("proposed_action", "none"))


def _run_agentic(
	cases: list[EvalConversation],
	provider: MockProvider | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	"""Run the real orchestrator against a fresh seeded database per scenario."""
	intents_correct = 0
	decisions_correct = 0
	actual_auto_resolutions = 0
	correct_escalations = 0
	expected_escalations = 0
	tool_attempts = 0
	tool_successes = 0
	turn_counts: list[int] = []
	latencies: list[float] = []
	final_actions_correct = 0
	case_results: list[dict[str, Any]] = []

	with tempfile.TemporaryDirectory(prefix="resolveai-eval-") as temporary_directory:
		for index, case in enumerate(cases, start=1):
			if provider is not None:
				provider.case = case
				provider.turn_index = 0
			database_path = Path(temporary_directory) / f"case-{index:02d}.db"
			database_url = f"sqlite:///{database_path.as_posix()}"
			seed_database(database_url)
			engine = create_engine(database_url, connect_args={"check_same_thread": False})
			conversation_id = f"EVAL-{index:02d}"
			final_response: dict[str, Any] = {}
			conversation_latency = 0.0
			turns_used = 0
			case_tool_attempts = 0
			case_tool_successes = 0
			with Session(engine) as session:
				for turn_index, message in enumerate(case.customer_turns, start=1):
					if provider is not None:
						provider.turn_index = turn_index
					started = time.perf_counter()
					final_response = handle_message(
						conversation_id=conversation_id,
						customer_id=case.customer_id,
						message=message,
						channel="evaluation",
						session=session,
					)
					conversation_latency += time.perf_counter() - started
					turns_used = turn_index
					state = final_response["state"]
					actions = state.get("actions")
					if actions is not None:
						case_tool_attempts += len(actions.actions_executed)
						case_tool_successes += sum(action.success for action in actions.actions_executed)
					escalation = state.get("escalation")
					if (
						(escalation is not None and escalation.decision == "escalate")
						or (actions is not None and actions.success)
					):
						break
				intent_result = state.get("intent")
				if intent_result is not None and intent_result.intent == case.expected_intent:
					intents_correct += 1
				escalation = state.get("escalation")
			actual_decision = escalation.decision if escalation is not None else "ask_more_info"
			if actual_decision == case.expected_decision:
				decisions_correct += 1
			if actual_decision == "auto_resolve":
				actual_auto_resolutions += 1
			if case.expected_decision == "escalate":
				expected_escalations += 1
				if actual_decision == "escalate":
					correct_escalations += 1
			actual_final_action = _actual_action(final_response.get("state", {}))
			if actual_final_action == case.expected_final_action:
				final_actions_correct += 1
			tool_attempts += case_tool_attempts
			tool_successes += case_tool_successes
			turn_counts.append(turns_used or len(case.customer_turns))
			latencies.append(conversation_latency)
			case_results.append(
				{
					"customer_id": case.customer_id,
					"expected_intent": case.expected_intent,
					"actual_intent": final_response.get("state", {}).get("intent").intent
					if final_response.get("state", {}).get("intent") is not None
					else "",
					"expected_decision": case.expected_decision,
					"actual_decision": actual_decision,
					"expected_final_action": case.expected_final_action,
					"actual_final_action": actual_final_action,
					"turns": turns_used,
					"latency_seconds": round(conversation_latency, 4),
					"tool_calls": case_tool_attempts,
					"successful_tool_calls": case_tool_successes,
				}
			)
			engine.dispose()

	count = len(cases)
	metrics = {
		"system": "Agentic ResolveAI",
		"conversations": count,
		"intent_accuracy": intents_correct / count if count else 0.0,
		"correct_decision_rate": decisions_correct / count if count else 0.0,
		"auto_resolution_rate": actual_auto_resolutions / count if count else 0.0,
		"correct_escalation_rate": (
			correct_escalations / expected_escalations if expected_escalations else 0.0
		),
		"tool_call_success_rate": tool_successes / tool_attempts if tool_attempts else 0.0,
		"avg_turns_to_resolution": mean(turn_counts) if turn_counts else 0.0,
		"avg_latency_seconds": mean(latencies) if latencies else 0.0,
		"final_action_accuracy": final_actions_correct / count if count else 0.0,
	}
	return metrics, case_results


def _run_baseline(cases: list[EvalConversation]) -> dict[str, Any]:
	"""Run the retrieval-only chatbot independently on each customer turn."""
	latencies: list[float] = []
	for case in cases:
		case_latency = 0.0
		for message in case.customer_turns:
			started = time.perf_counter()
			baseline_rag.answer_query(message)
			case_latency += time.perf_counter() - started
		latencies.append(case_latency)
	return {
		"system": "Baseline RAG",
		"conversations": len(cases),
		"intent_accuracy": None,
		"correct_decision_rate": None,
		"auto_resolution_rate": None,
		"correct_escalation_rate": None,
		"tool_call_success_rate": None,
		"avg_turns_to_resolution": 1.0,
		"avg_latency_seconds": mean(latencies) if latencies else 0.0,
		"final_action_accuracy": None,
	}


def _format_metric(value: Any, percentage: bool = False) -> str:
	"""Format one metric for CSV, terminal, and Markdown output."""
	if value is None:
		return "N/A"
	if percentage:
		return f"{float(value):.1%}"
	return f"{float(value):.3f}"


def _write_results(agentic: dict[str, Any], baseline: dict[str, Any]) -> None:
	"""Write metrics to CSV/Markdown and generate a PNG comparison chart."""
	rows = [agentic, baseline]
	fieldnames = list(agentic.keys())
	with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	percentage_keys = {
		"intent_accuracy", "correct_decision_rate", "auto_resolution_rate",
		"correct_escalation_rate", "tool_call_success_rate", "final_action_accuracy",
	}
	metrics_table = [
		("Intent accuracy", "intent_accuracy"),
		("Correct decision rate", "correct_decision_rate"),
		("Auto-resolution rate", "auto_resolution_rate"),
		("Correct-escalation rate", "correct_escalation_rate"),
		("Tool-call success rate", "tool_call_success_rate"),
		("Final-action accuracy", "final_action_accuracy"),
		("Avg turns to resolution", "avg_turns_to_resolution"),
		("Avg latency (s)", "avg_latency_seconds"),
	]
	markdown = [
		"# ResolveAI Evaluation Results",
		"",
		f"Conversations evaluated: {agentic['conversations']}",
		"",
		"| Metric | Agentic ResolveAI | Baseline RAG |",
		"|---|---:|---:|",
	]
	for label, key in metrics_table:
		markdown.append(
			f"| {label} | {_format_metric(agentic[key], key in percentage_keys)} | "
			f"{_format_metric(baseline[key], key in percentage_keys)} |"
		)
	RESULTS_MARKDOWN_PATH.write_text("\n".join(markdown) + "\n", encoding="utf-8")

	import matplotlib.pyplot as plt

	labels = [agentic["system"], baseline["system"]]
	figure, axes = plt.subplots(1, 2, figsize=(9, 4))
	for axis, metric_key, title, ylabel in (
		(axes[0], "avg_turns_to_resolution", "Average turns", "Turns"),
		(axes[1], "avg_latency_seconds", "Average latency", "Seconds"),
	):
		values = [float(agentic[metric_key]), float(baseline[metric_key])]
		axis.bar(labels, values, color=["#d4513b", "#417b76"])
		axis.set_title(title)
		axis.set_ylabel(ylabel)
		axis.tick_params(axis="x", rotation=12)
	figure.suptitle("ResolveAI: Agentic vs. Baseline RAG")
	figure.tight_layout()
	figure.savefig(RESULTS_CHART_PATH, dpi=160)
	plt.close(figure)

	print("| Metric | Agentic ResolveAI | Baseline RAG |")
	print("|---|---:|---:|")
	for label, key in metrics_table:
		print(
			f"| {label} | {_format_metric(agentic[key], key in percentage_keys)} | "
			f"{_format_metric(baseline[key], key in percentage_keys)} |"
		)
	print(f"Saved {RESULTS_CSV_PATH}, {RESULTS_MARKDOWN_PATH}, and {RESULTS_CHART_PATH}.")


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[EvalConversation]:
	"""Load and validate the 40 scripted evaluation conversations."""
	data = json.loads(path.read_text(encoding="utf-8"))
	cases = [EvalConversation.model_validate(item) for item in data]
	if len(cases) != 40:
		raise ValueError(f"Expected 40 scripted conversations, found {len(cases)}.")
	if sum(len(case.customer_turns) > 1 for case in cases) < 8:
		raise ValueError("The evaluation set must contain at least eight multi-turn conversations.")
	if sum(case.expected_decision == "escalate" for case in cases) != 6:
		raise ValueError("The evaluation set must contain exactly six escalations.")
	if sum(case.policy_blocked for case in cases) != 4:
		raise ValueError("The evaluation set must contain exactly four policy-blocked cases.")
	return cases


def main() -> None:
	"""Run both systems and save aggregate evaluation metrics and chart."""
	parser = argparse.ArgumentParser(description="Evaluate ResolveAI against a simple RAG baseline.")
	parser.add_argument("--mock", action="store_true", help="Use deterministic fake LLM responses offline.")
	parser.add_argument(
		"--delay",
		type=float,
		default=0.0,
		metavar="SECONDS",
		help="Pause before each live Gemini call to reduce rate-limit pressure.",
	)
	arguments = parser.parse_args()
	cases = load_scenarios()
	provider = MockProvider() if arguments.mock else None
	if provider is not None:
		_install_mock_provider(provider)
	else:
		_install_delay(max(0.0, arguments.delay))
	agentic_metrics, _ = _run_agentic(cases, provider)
	baseline_metrics = _run_baseline(cases)
	_write_results(agentic_metrics, baseline_metrics)


if __name__ == "__main__":
	main()