"""Draft, embed, and review knowledge articles from resolved support tickets."""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import Ticket
from db.session import SessionLocal
from kb.ingest import ARTICLES_DIRECTORY, ingest_kb
from llm.client import call_llm_json

KB_DRAFT_PROMPT = """Draft a concise reusable FAQ article based only on the resolved support
ticket facts. Include a useful title, category, one valid workflow_id, and clear numbered
resolution steps. Remove all customer names, emails, phone numbers, and identifying data.
Use only the supplied investigation and successful actions; do not invent policy facts."""
VALID_WORKFLOWS = {
	"wrong_item", "damaged_item", "payment_deducted", "order_status", "cancel_order",
	"account_locked", "password_reset", "subscription_sync", "refund_request",
	"fraud_escalate", "legal_escalate", "general_info",
}
PENDING_DIRECTORY = ARTICLES_DIRECTORY / "auto"
logger = logging.getLogger(__name__)


class KBDraft(BaseModel):
	"""Structured FAQ draft returned by Gemini."""

	title: str
	category: str
	workflow_id: str
	body: str


class PendingKBArticle(BaseModel):
	"""Article metadata shown in the supervisor review panel."""

	article_id: str
	title: str
	category: str
	reliability_score: float
	workflow_id: str
	status: Literal["pending_review", "approved", "rejected"]
	body: str


class KBReviewResult(BaseModel):
	"""Review decision and whether the local vector index was refreshed."""

	article_id: str
	status: Literal["approved", "rejected"]
	embedded: bool
	embedding_error: str | None = None


@contextmanager
def _using_session(session: Session | None) -> Iterator[Session]:
	"""Use a supplied database session or open a managed session."""
	if session is not None:
		yield session
	else:
		with SessionLocal() as managed_session:
			yield managed_session


def _slug(text: str) -> str:
	"""Convert a title to a filesystem-safe lowercase slug."""
	return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-") or "support-faq"


def _fallback_draft(ticket: Ticket) -> KBDraft:
	"""Create a privacy-safe article outline if Gemini drafting fails."""
	workflow = ticket.intent if ticket.intent in VALID_WORKFLOWS else "general_info"
	body = (
		"## Resolution steps\n\n"
		"1. Confirm the customer's issue and identify the affected support workflow.\n"
		"2. Review the relevant account, order, or payment records before taking action.\n"
		"3. Apply the documented policy and verify the outcome with the customer.\n\n"
		f"## Investigation pattern\n\n{ticket.investigation}"
	)
	return KBDraft(
		title=f"Support FAQ: {ticket.category.replace('_', ' ').title()}",
		category=ticket.category,
		workflow_id=workflow,
		body=body,
	)


def _write_article(path: Path, draft: KBDraft, status: str) -> None:
	"""Write an article with structured YAML front matter and safe defaults."""
	path.parent.mkdir(parents=True, exist_ok=True)
	workflow_id = draft.workflow_id if draft.workflow_id in VALID_WORKFLOWS else "general_info"
	body = draft.body.strip()
	if not re.search(r"(?m)^\s*1[.)]\s", body):
		body = "## Resolution steps\n\n1. Review the documented issue.\n2. Verify all relevant records.\n3. Confirm the outcome with the customer.\n\n" + body
	metadata = {
		"title": draft.title.strip() or "Support FAQ",
		"category": draft.category.strip() or "general",
		"reliability_score": 0.6,
		"workflow_id": workflow_id,
		"status": status,
	}
	front_matter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False).strip()
	path.write_text(f"---\n{front_matter}\n---\n\n{body}\n", encoding="utf-8")


def draft_faq(ticket_id: int, session: Session | None = None) -> PendingKBArticle:
	"""Draft a pending-review FAQ from a resolved ticket and refresh KB embeddings."""
	with _using_session(session) as db:
		ticket = db.get(Ticket, ticket_id)
		if ticket is None:
			raise ValueError(f"Ticket {ticket_id} was not found.")
		if ticket.status not in {"resolved", "closed"}:
			raise ValueError(f"Ticket {ticket_id} must be resolved before KB expansion.")
		facts = (
			f"Intent: {ticket.intent}\nCategory: {ticket.category}\n"
			f"Issue summary: {ticket.summary}\nInvestigation: {ticket.investigation}\n"
			f"Successful actions: {[action for action in ticket.actions_taken if action.get('success')]}"
		)
		try:
			draft = call_llm_json(KB_DRAFT_PROMPT, facts, KBDraft, "kb_expansion")
		except Exception as error:
			logger.warning("KB article draft failed; using ticket-grounded template: %s", error)
			draft = _fallback_draft(ticket)
		article_id = f"ticket-{ticket.id}-{_slug(draft.title)}"
		article_path = PENDING_DIRECTORY / f"{article_id}.md"
		_write_article(article_path, draft, "pending_review")

	embedded = False
	embedding_error = None
	try:
		ingest_kb()
		embedded = True
	except Exception as error:
		embedding_error = str(error)
		logger.exception("Saved pending FAQ %s but failed to refresh embeddings", article_id)
	return PendingKBArticle(
		article_id=article_id,
		title=draft.title,
		category=draft.category,
		reliability_score=0.6,
		workflow_id=draft.workflow_id if draft.workflow_id in VALID_WORKFLOWS else "general_info",
		status="pending_review",
		body=draft.body,
	)


def list_pending_articles() -> list[PendingKBArticle]:
	"""Read pending-review articles from the auto-generated KB directory."""
	articles: list[PendingKBArticle] = []
	if not PENDING_DIRECTORY.exists():
		return articles
	for path in sorted(PENDING_DIRECTORY.glob("*.md")):
		parts = path.read_text(encoding="utf-8").split("---", 2)
		if len(parts) != 3:
			continue
		metadata = yaml.safe_load(parts[1]) or {}
		if metadata.get("status") != "pending_review":
			continue
		articles.append(
			PendingKBArticle(
				article_id=path.stem,
				title=str(metadata.get("title", path.stem)),
				category=str(metadata.get("category", "general")),
				reliability_score=float(metadata.get("reliability_score", 0.6)),
				workflow_id=str(metadata.get("workflow_id", "general_info")),
				status="pending_review",
				body=parts[2].strip(),
			)
		)
	return articles


def review_article(article_id: str, decision: Literal["approve", "reject"]) -> KBReviewResult:
	"""Approve or reject an auto article and rebuild the local vector index."""
	if not re.fullmatch(r"[a-z0-9-]+", article_id):
		raise ValueError("Invalid KB article ID.")
	article_path = PENDING_DIRECTORY / f"{article_id}.md"
	if not article_path.is_file():
		raise ValueError(f"KB article {article_id} was not found.")
	parts = article_path.read_text(encoding="utf-8").split("---", 2)
	if len(parts) != 3:
		raise ValueError(f"KB article {article_id} has invalid front matter.")
	metadata = yaml.safe_load(parts[1]) or {}
	if metadata.get("status") != "pending_review":
		raise ValueError(f"KB article {article_id} is not pending review.")
	status = "approved" if decision == "approve" else "rejected"
	metadata["status"] = status
	front_matter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False).strip()
	article_path.write_text(f"---\n{front_matter}\n---\n\n{parts[2].strip()}\n", encoding="utf-8")
	embedded = False
	embedding_error = None
	try:
		ingest_kb()
		embedded = True
	except Exception as error:
		embedding_error = str(error)
		logger.exception("Could not refresh KB embeddings after reviewing %s", article_id)
	return KBReviewResult(
		article_id=article_id,
		status=status,
		embedded=embedded,
		embedding_error=embedding_error,
	)