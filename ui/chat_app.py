"""ResolveAI customer chat and live agent trace, backed by the FastAPI service."""

import os
from uuid import uuid4

import requests
import streamlit as st

from ui.styles import apply_styles, render_typing_indicator
from ui.theme import ASSISTANT_AVATAR, USER_AVATAR

DEFAULT_API_URL = os.getenv("RESOLVEAI_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 10
DEMO_SCENARIOS = {
	"c001_wrong_item": {
		"label": "C001 · Wrong item",
		"customer_id": "C001",
		"conversation_id": "DEMO-C001-WRONG",
		"turns": ["I received the wrong item for ORD10001."],
	},
	"c002_payment_failure": {
		"label": "C002 · Payment failed",
		"customer_id": "C002",
		"conversation_id": "DEMO-C002-PAYMENT",
		"turns": [
			"My payment failed, but the amount was deducted.",
			"The order is ORD10002.",
		],
	},
	"c008_fraud": {
		"label": "C008 · Fraud complaint",
		"customer_id": "C008",
		"conversation_id": "DEMO-C008-FRAUD",
		"turns": ["I see an unauthorized charge I do not recognize."],
	},
}


def _get_customers(api_url: str) -> list[dict[str, str]]:
	"""Fetch customer options from the ResolveAI API."""
	response = requests.get(f"{api_url}/customers", timeout=REQUEST_TIMEOUT_SECONDS)
	response.raise_for_status()
	return response.json()


def _reset_conversation(customer_id: str) -> None:
	"""Initialize session state for a selected customer conversation."""
	st.session_state.customer_id = customer_id
	st.session_state.conversation_id = f"conv-{uuid4()}"
	st.session_state.chat_history = []
	st.session_state.trace = []
	st.session_state.ticket = None
	st.session_state.awaiting_csat = False


def _keep_customer_trace(trace: object) -> None:
	"""Keep the latest full agent trace and ignore rating or system-only turns."""
	if not isinstance(trace, list) or not trace:
		return
	customer_steps = [
		entry
		for entry in trace
		if isinstance(entry, dict)
		and str(entry.get("agent", "")).lower()
		not in {"csat_prompt", "system", "system_message"}
	]
	if customer_steps:
		st.session_state["trace"] = customer_steps


def _read_chat_response(response: requests.Response) -> dict[str, object]:
	"""Persist the customer-agent trace first, then log the response."""
	response.raise_for_status()
	payload = response.json()
	_keep_customer_trace(payload.get("trace"))
	try:
		print(f"/chat trace steps: {len(payload.get('trace', []))}", flush=True)
	except Exception:
		pass
	return payload


def _render_trace() -> None:
	"""Render every agent step, its summary, and the total execution time."""
	trace = st.session_state.get("trace", [])
	if not trace:
		st.caption("Agent steps will appear here after a message.")
		return
	total_duration = 0
	for index, entry in enumerate(trace):
		agent = str(entry.get("agent", "agent"))
		duration = int(entry.get("duration_ms", 0))
		total_duration += duration
		with st.container(border=True):
			name_column, duration_column = st.columns([4, 1], vertical_alignment="center")
			with name_column:
				st.markdown(
					f'<div class="resolveai-trace-step">'
					f'{agent.replace("_", " ").title()}</div>',
					unsafe_allow_html=True,
				)
			with duration_column:
				st.markdown(
					f'<div class="resolveai-trace-duration">{duration} ms</div>',
					unsafe_allow_html=True,
				)
			st.caption(str(entry.get("output_summary", "Step completed.")))
			with st.expander("Step details", expanded=index == len(trace) - 1):
				st.caption(f"Confidence: {float(entry.get('confidence', 0.0)):.2f}")
				st.markdown("**Input**")
				st.write(entry.get("input_summary", ""))
				st.markdown("**Output**")
				st.write(entry.get("output_summary", ""))
				tools = entry.get("tools_called", [])
				st.markdown("**Tools**")
				st.write(", ".join(str(tool) for tool in tools) if tools else "None")
	st.markdown(f"**Total time · {total_duration} ms**")


def _render_ticket(ticket: dict[str, object] | None) -> None:
	"""Show an escalated ticket in a compact, readable ticket panel."""
	if ticket is None:
		return
	with st.container(border=True):
		st.markdown(f"**Ticket #{ticket.get('ticket_id', ticket.get('id', ''))}**")
		st.caption(
			f"{ticket.get('status', 'opened').replace('_', ' ').title()} · "
			f"{ticket.get('priority', 'medium').title()} priority"
		)
		st.write(ticket.get("summary", "Support ticket created."))
		if ticket.get("escalation_reason"):
			st.info(str(ticket["escalation_reason"]))


def _send_csat(api_url: str, rating: int, comment: str) -> None:
	"""Submit a CSAT response and update the local conversation state."""
	response = requests.post(
		f"{api_url}/csat",
		json={
			"conversation_id": st.session_state.conversation_id,
			"rating": rating,
			"comment": comment or None,
		},
		timeout=REQUEST_TIMEOUT_SECONDS,
	)
	response.raise_for_status()
	st.session_state.awaiting_csat = False
	st.session_state.chat_history.append(
		{"role": "assistant", "content": "Thanks for sharing your feedback."}
	)


def _run_demo_scenario(api_url: str, scenario_key: str) -> None:
	"""Run one scripted scenario by making normal HTTP chat requests."""
	scenario = DEMO_SCENARIOS[scenario_key]
	st.session_state.customer_id = scenario["customer_id"]
	st.session_state.conversation_id = f"{scenario['conversation_id']}-{uuid4().hex[:6]}"
	st.session_state.chat_history = []
	st.session_state.ticket = None
	st.session_state.awaiting_csat = False
	for message in scenario["turns"]:
		st.session_state.chat_history.append({"role": "user", "content": message})
		with st.chat_message("user", avatar=USER_AVATAR):
			st.markdown(message)
		with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
			render_typing_indicator()
			with st.status("Understanding your issue...", expanded=False) as status:
				status.update(label="Searching knowledge base...")
				response = requests.post(
					f"{api_url}/chat",
					json={
						"conversation_id": st.session_state.conversation_id,
						"customer_id": scenario["customer_id"],
						"message": message,
						"channel": "streamlit_demo",
					},
					timeout=120,
				)
				status.update(label="Preparing reply...")
				payload = _read_chat_response(response)
		st.session_state.chat_history.append(
			{"role": "assistant", "content": payload["reply"]}
		)
		st.session_state.ticket = payload.get("ticket") or st.session_state.ticket
		st.session_state.awaiting_csat = bool(
			payload.get("state", {}).get("awaiting_csat")
		)


def main() -> None:
	"""Render the ResolveAI chat application and call the backend over HTTP."""
	st.set_page_config(page_title="ResolveAI | Customer Support", page_icon="R", layout="wide")
	apply_styles()
	st.sidebar.caption(f"DEBUG trace steps: {len(st.session_state.get('trace', []))}")
	st.session_state.setdefault("is_loading", False)
	connection_error: requests.RequestException | None = None
	customers: list[dict[str, str]] = []
	with st.sidebar:
		st.markdown('<div class="resolveai-sidebar-label">CONNECTION</div>', unsafe_allow_html=True)
		api_url = st.text_input("API URL", DEFAULT_API_URL).rstrip("/")
		try:
			customers = _get_customers(api_url)
		except requests.RequestException as error:
			connection_error = error
		customer_by_id = {customer["id"]: customer for customer in customers}
		pending_customer_id = st.session_state.pop("pending_demo_customer_id", None)
		pending_demo_key = st.session_state.pop("pending_demo_key", None)
		if customers:
			if pending_customer_id in customer_by_id:
				st.session_state["customer_selector"] = pending_customer_id
			selected_customer_id = st.selectbox(
				"Customer",
				list(customer_by_id),
				format_func=lambda customer_id: (
					f"{customer_by_id[customer_id]['name']} · "
					f"{customer_by_id[customer_id]['tier'].title()}"
				),
				key="customer_selector",
			)
			if (
				st.session_state.get("customer_id") != selected_customer_id
				and pending_demo_key is None
			):
				_reset_conversation(selected_customer_id)
		st.markdown('<div class="resolveai-sidebar-label">DEMO SCENARIOS</div>', unsafe_allow_html=True)
		for scenario_key, scenario in DEMO_SCENARIOS.items():
			if st.button(
				scenario["label"],
				key=f"demo-{scenario_key}",
				use_container_width=True,
				disabled=st.session_state.is_loading or not customers,
			):
				st.session_state["pending_demo_key"] = scenario_key
				st.session_state["pending_demo_customer_id"] = scenario["customer_id"]
				st.session_state.is_loading = True
				st.rerun()

	status_class = "offline" if connection_error else "online"
	status_label = "API disconnected" if connection_error else "API connected"
	st.markdown(
		'<div class="resolveai-page-header">'
		'<div class="resolveai-page-title">'
		'<h1>ResolveAI</h1><span class="resolveai-status '
		f'{status_class}">{status_label}</span></div>'
		'<p>Customer support, made clearer.</p>'
		"</div>",
		unsafe_allow_html=True,
	)
	if connection_error:
		st.error(f"Could not reach ResolveAI API: {connection_error}")
		return
	if not customers:
		st.warning("No customers are available.")
		return

	if pending_demo_key is not None:
		try:
			_run_demo_scenario(api_url, pending_demo_key)
		except requests.RequestException as error:
			st.session_state["demo_error"] = str(error)
		finally:
			st.session_state.is_loading = False
		st.rerun()

	pending_prompt = st.session_state.pop("pending_chat_message", None)
	if pending_prompt is not None:
		try:
			with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
				render_typing_indicator()
				with st.status("Understanding your issue...", expanded=False) as status:
					status.update(label="Searching knowledge base...")
					response = requests.post(
						f"{api_url}/chat",
						json={
							"conversation_id": st.session_state.conversation_id,
							"customer_id": st.session_state.customer_id,
							"message": pending_prompt,
							"channel": "streamlit",
						},
						timeout=120,
					)
					status.update(label="Preparing reply...")
					payload = _read_chat_response(response)
			st.session_state.chat_history.append(
				{"role": "assistant", "content": payload["reply"]}
			)
			st.session_state.ticket = payload.get("ticket")
			st.session_state.awaiting_csat = bool(
				payload.get("state", {}).get("awaiting_csat")
			)
		except requests.RequestException as error:
			st.session_state["chat_error"] = str(error)
		finally:
			st.session_state.is_loading = False
		st.rerun()

	chat_column, trace_column = st.columns([1.65, 1], gap="large")
	with chat_column:
		st.subheader("Conversation")
		chat_history = st.session_state.get("chat_history", [])
		if not chat_history:
			st.markdown(
				'<div class="resolveai-welcome">'
				'<h3>How can we help today?</h3>'
				'<p>Ask a question or choose a suggestion to get started.</p>'
				"</div>",
				unsafe_allow_html=True,
			)
			suggestions = [
				"Where is my order?",
				"My payment was deducted but the order failed.",
				"I need help with an unauthorized charge.",
			]
			suggestion_columns = st.columns(3, gap="small")
			for index, suggestion in enumerate(suggestions):
				if suggestion_columns[index].button(
					suggestion, key=f"suggested-question-{index}", use_container_width=True
				):
					st.session_state.chat_history.append(
						{"role": "user", "content": suggestion}
					)
					st.session_state["pending_chat_message"] = suggestion
					st.session_state.is_loading = True
					st.rerun()
		for item in chat_history:
			avatar = USER_AVATAR if item["role"] == "user" else ASSISTANT_AVATAR
			with st.chat_message(item["role"], avatar=avatar):
				st.markdown(item["content"])
		if demo_error := st.session_state.pop("demo_error", None):
			st.error(f"Demo scenario could not be run: {demo_error}")
		if chat_error := st.session_state.pop("chat_error", None):
			st.error(f"Message could not be sent: {chat_error}")
		if st.session_state.get("awaiting_csat"):
			with st.form("resolveai-csat", border=True):
				st.markdown("### How was your support experience?")
				rating = st.feedback("stars", key="csat-rating")
				comment = st.text_input("Comment (optional)")
				submitted = st.form_submit_button(
					"Submit rating", disabled=rating is None, type="primary"
				)
			if submitted:
				try:
					_send_csat(api_url, rating + 1, comment)
					st.rerun()
				except requests.RequestException as error:
					st.error(f"Could not submit feedback: {error}")
		prompt = st.chat_input(
			"Message ResolveAI", disabled=st.session_state.is_loading
		)
		if prompt:
			st.session_state.chat_history.append({"role": "user", "content": prompt})
			st.session_state["pending_chat_message"] = prompt
			st.session_state.is_loading = True
			st.rerun()
		_render_ticket(st.session_state.get("ticket"))
	with trace_column:
		trace = st.session_state.get("trace", [])
		trace_height: int | str = 600 if len(trace) > 7 else "content"
		with st.container(height=trace_height, border=True):
			st.markdown(
				'<div class="resolveai-trace-heading">Agent Trace</div>',
				unsafe_allow_html=True,
			)
			_render_trace()


if __name__ == "__main__":
	main()