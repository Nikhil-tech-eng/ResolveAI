"""Pre-record Gemini responses for the three ResolveAI demonstration scenarios."""

import json
import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def record_demo_cache() -> int:
	"""Run C001, C002, and C008 against a temporary seeded database and record calls."""
	load_dotenv(PROJECT_ROOT / ".env")
	if not os.getenv("GEMINI_API_KEY"):
		raise RuntimeError("Set GEMINI_API_KEY in .env before recording demo responses.")
	os.environ["DEMO_MODE"] = "false"
	os.environ["DEMO_CACHE_RECORD"] = "true"

	from agents import knowledge_agent
	from agents.orchestrator import handle_message
	from data.seed_data import seed_database
	from kb.search import _lexical_search
	from llm.client import DEMO_CACHE_PATH

	logging.basicConfig(level=logging.INFO)
	try:
		previous_cache = json.loads(DEMO_CACHE_PATH.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		previous_cache = {}
	
	knowledge_agent.search_kb = lambda query, category=None, k=4: _lexical_search(query, category, k)
	with tempfile.TemporaryDirectory(prefix="resolveai-demo-cache-") as temporary_directory:
		database_path = Path(temporary_directory) / "support.db"
		database_url = f"sqlite:///{database_path.as_posix()}"
		seed_database(database_url)
		engine = create_engine(database_url, connect_args={"check_same_thread": False})
		with Session(engine) as session:
			handle_message(
				"DEMO-C001-WRONG",
				"C001",
				"I received the wrong item for ORD10001.",
				"demo",
				session=session,
			)
			handle_message(
				"DEMO-C002-PAYMENT",
				"C002",
				"My payment failed, but the amount was deducted.",
				"demo",
				session=session,
			)
			handle_message(
				"DEMO-C002-PAYMENT",
				"C002",
				"The order is ORD10002.",
				"demo",
				session=session,
			)
			handle_message(
				"DEMO-C008-FRAUD",
				"C008",
				"I see an unauthorized charge I do not recognize.",
				"demo",
				session=session,
			)
		engine.dispose()

	cache = json.loads(DEMO_CACHE_PATH.read_text(encoding="utf-8"))
	new_entries = len(set(cache) - set(previous_cache))
	if not cache:
		raise RuntimeError("Demo cache is empty; check Gemini output and cache path.")
	print(
		f"Demo cache ready with {len(cache)} responses; "
		f"{new_entries} new keys recorded in {DEMO_CACHE_PATH}."
	)
	return len(cache)


def main() -> None:
	"""Run the response-cache recording workflow."""
	record_demo_cache()


if __name__ == "__main__":
	main()