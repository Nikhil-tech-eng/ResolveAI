"""ResolveAI supervisor dashboard backed by the FastAPI metrics endpoints."""

import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from ui.styles import apply_styles
from ui.theme import AMBER, BG, CARD, GRID, MUTED, SAGE, SILVER, TEXT, WALNUT

DEFAULT_API_URL = os.getenv("RESOLVEAI_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 10
TICKET_TRANSITIONS = {
	"opened": ["assigned"],
	"assigned": ["pending_customer"],
	"pending_customer": ["assigned", "resolved"],
	"resolved": ["opened", "closed"],
	"closed": [],
}


@st.cache_data(ttl=30)
def _get_json(api_url: str, path: str) -> object:
	"""Fetch a JSON resource from the ResolveAI API."""
	started_at = time.perf_counter()
	try:
		response = requests.get(f"{api_url}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
		response.raise_for_status()
		return response.json()
	finally:
		elapsed = time.perf_counter() - started_at
		print(f"GET {path} took {elapsed:.3f} seconds")


def _render_kpis(metrics: dict[str, object]) -> None:
	"""Render the five supervisor KPI values in a compact row."""
	columns = st.columns(5)
	columns[0].metric("Open tickets", int(metrics.get("open_tickets", 0)))
	columns[1].metric("Resolution rate", f"{float(metrics.get('resolution_rate', 0.0)):.1%}")
	average_minutes = metrics.get("avg_resolution_time_minutes")
	columns[2].metric(
		"Avg resolution time",
		f"{float(average_minutes):.1f} min" if average_minutes is not None else "—",
	)
	columns[3].metric("Escalation rate", f"{float(metrics.get('escalation_rate', 0.0)):.1%}")
	average_csat = metrics.get("avg_csat")
	columns[4].metric("Avg CSAT", f"{float(average_csat):.2f} / 5" if average_csat is not None else "—")


def _render_charts(metrics: dict[str, object]) -> None:
	"""Render intent distribution and daily ticket/escalation charts."""
	left, right = st.columns(2, gap="large")
	intents = metrics.get("intent_distribution", [])
	with left:
		st.subheader("Intent distribution")
		if intents:
			intent_frame = pd.DataFrame(intents)
			figure = px.bar(
				intent_frame,
				x="intent",
				y="count",
				color="intent",
				color_discrete_sequence=[SAGE, AMBER, WALNUT, MUTED],
			)
			figure.update_layout(
				showlegend=False,
				xaxis_title=None,
				yaxis_title="Tickets",
				paper_bgcolor=BG,
				plot_bgcolor=CARD,
				font={"color": TEXT, "family": "Inter, sans-serif"},
				xaxis={"gridcolor": GRID, "linecolor": SILVER},
				yaxis={"gridcolor": GRID, "linecolor": SILVER},
			)
			st.plotly_chart(figure, use_container_width=True)
		else:
			st.caption("No ticket intents have been recorded.")
	with right:
		st.subheader("Tickets over time")
		timeline = metrics.get("tickets_over_time", [])
		escalations = metrics.get("escalations_over_time", [])
		if timeline:
			ticket_frame = pd.DataFrame(timeline).rename(columns={"count": "Tickets"})
			ticket_frame["Series"] = "Tickets"
			if escalations:
				escalation_frame = pd.DataFrame(escalations).rename(columns={"count": "Escalations"})
				escalation_frame = escalation_frame.rename(columns={"Escalations": "Tickets"})
				escalation_frame["Series"] = "Escalations"
				chart_frame = pd.concat([ticket_frame, escalation_frame], ignore_index=True)
			else:
				chart_frame = ticket_frame
			figure = px.line(
				chart_frame,
				x="date",
				y="Tickets",
				color="Series",
				markers=True,
				color_discrete_map={"Tickets": SAGE, "Escalations": AMBER},
			)
			figure.update_layout(
				xaxis_title=None,
				yaxis_title="Count",
				paper_bgcolor=BG,
				plot_bgcolor=CARD,
				font={"color": TEXT, "family": "Inter, sans-serif"},
				xaxis={"gridcolor": GRID, "linecolor": SILVER},
				yaxis={"gridcolor": GRID, "linecolor": SILVER},
			)
			st.plotly_chart(figure, use_container_width=True)
		else:
			st.caption("No tickets have been recorded over time.")


def _render_ticket_controls(api_url: str, tickets: list[dict[str, object]]) -> None:
	"""Filter ticket rows and expose lifecycle transition controls."""
	st.subheader("Tickets")
	statuses = ["all", "opened", "assigned", "pending_customer", "resolved", "closed"]
	status_filter = st.selectbox("Status filter", statuses)
	filtered = [
		ticket
		for ticket in tickets
		if status_filter == "all" or ticket.get("status") == status_filter
	]
	if filtered:
		table_rows = [
			{
				"ID": ticket.get("id"),
				"Customer": ticket.get("customer_name"),
				"Tier": ticket.get("customer_tier"),
				"Intent": ticket.get("intent"),
				"Priority": ticket.get("priority"),
				"Status": ticket.get("status"),
				"SLA risk": ticket.get("sla_risk", False),
				"Summary": ticket.get("summary"),
			}
			for ticket in filtered
		]
		st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
		options = {f"#{ticket['id']} · {ticket['customer_name']} · {ticket['status']}": ticket for ticket in filtered}
		with st.form("ticket-status-change"):
			selected_label = st.selectbox("Ticket", list(options))
			selected_ticket = options[selected_label]
			current_status = str(selected_ticket["status"])
			allowed = TICKET_TRANSITIONS.get(current_status, [])
			if allowed:
				new_status = st.selectbox("Change status to", allowed)
				actor = st.text_input("Actor", value="supervisor")
				submitted = st.form_submit_button("Update ticket")
			else:
				st.caption("Closed tickets have no available transitions.")
				submitted = False
				new_status = "closed"
				actor = "supervisor"
		if submitted:
			try:
				response = requests.patch(
					f"{api_url}/tickets/{selected_ticket['id']}",
					json={"status": new_status, "actor": actor},
					timeout=REQUEST_TIMEOUT_SECONDS,
				)
				response.raise_for_status()
				st.success(f"Ticket #{selected_ticket['id']} moved to {new_status}.")
				st.rerun()
			except requests.RequestException as error:
				st.error(f"Ticket update failed: {error}")
	else:
		st.caption("No tickets match this status.")


def main() -> None:
	"""Render the ResolveAI supervisor dashboard using live API data."""
	st.set_page_config(page_title="ResolveAI | Supervisor", layout="wide")
	apply_styles()
	title_column, refresh_column = st.columns([5, 1], vertical_alignment="center")
	with title_column:
		st.title("ResolveAI | Supervisor")
	with refresh_column:
		if st.button("Refresh", use_container_width=True):
			_get_json.clear()
			st.rerun()
	with st.sidebar:
		api_url = st.text_input("API URL", DEFAULT_API_URL).rstrip("/")
	try:
		with st.spinner("Loading dashboard..."):
			metrics = _get_json(api_url, "/metrics")
			tickets = _get_json(api_url, "/tickets")
			alerts = _get_json(api_url, "/alerts")
			pending_articles = _get_json(api_url, "/kb/pending")
	except requests.RequestException as error:
		st.error(f"Could not load ResolveAI dashboard data: {error}")
		return
	if (
		not isinstance(metrics, dict)
		or not isinstance(tickets, list)
		or not isinstance(alerts, list)
		or not isinstance(pending_articles, list)
	):
		st.error("ResolveAI API returned an unexpected dashboard response.")
		return

	_render_kpis(metrics)
	_render_charts(metrics)
	alert_column, simulation_column = st.columns([4, 1], vertical_alignment="center")
	with alert_column:
		st.subheader("Root-cause alerts")
	with simulation_column:
		if st.button("Simulate outage", use_container_width=True):
			try:
				response = requests.post(
					f"{api_url}/alerts/simulate-outage",
					timeout=REQUEST_TIMEOUT_SECONDS,
				)
				response.raise_for_status()
				st.success("Injected 20 payment-failure conversations.")
				st.rerun()
			except requests.RequestException as error:
				st.error(f"Outage simulation failed: {error}")
	if alerts:
		for alert in alerts:
			st.error(alert.get("message", "Root-cause spike detected."))
	else:
		st.caption("No root-cause spikes detected in the last 30 minutes.")

	st.subheader("Pending KB Articles")
	if pending_articles:
		for article in pending_articles:
			article_id = str(article["article_id"])
			with st.expander(str(article.get("title", article_id))):
				st.caption(
					f"{article.get('category', 'general')} · "
					f"{article.get('workflow_id', 'general_info')} · "
					f"Reliability {float(article.get('reliability_score', 0.6)):.1f}"
				)
				st.markdown(str(article.get("body", "")))
				approve_column, reject_column = st.columns(2)
				for column, decision, label in (
					(approve_column, "approve", "Approve"),
					(reject_column, "reject", "Reject"),
				):
					with column:
						if st.button(label, key=f"{decision}-{article_id}", use_container_width=True):
							try:
								response = requests.post(
									f"{api_url}/kb/articles/{article_id}/review",
									json={"decision": decision},
									timeout=120,
								)
								response.raise_for_status()
								status_label = "approved" if decision == "approve" else "rejected"
								st.success(f"Article {status_label}.")
								st.rerun()
							except requests.RequestException as error:
								st.error(f"Article review failed: {error}")
	else:
		st.caption("No generated articles are awaiting review.")

	_at_risk = [ticket for ticket in tickets if ticket.get("sla_risk")]
	st.subheader("SLA at risk")
	if _at_risk:
		st.dataframe(
			pd.DataFrame(
				[
					{
						"Ticket": ticket.get("id"),
						"Customer": ticket.get("customer_name"),
						"Deadline": ticket.get("sla_deadline"),
						"Priority": ticket.get("priority"),
					}
					for ticket in _at_risk
				]
			),
			hide_index=True,
			use_container_width=True,
		)
	else:
		st.caption("No tickets are currently marked at risk.")
	_render_ticket_controls(api_url, tickets)


if __name__ == "__main__":
	main()