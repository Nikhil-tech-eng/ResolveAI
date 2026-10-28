"""Telegram adapter that links chats to customers and forwards to ResolveAI."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from sqlalchemy import func, select

from db.models import Customer
from db.session import SessionLocal

load_dotenv()

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINKS_PATH = PROJECT_ROOT / "data" / "telegram_links.json"
API_URL = os.getenv("RESOLVEAI_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30


def _load_links() -> dict[str, str]:
	"""Load chat ID to customer ID links from the local JSON file."""
	if not LINKS_PATH.exists():
		return {}
	try:
		value = json.loads(LINKS_PATH.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as error:
		logger.warning("Could not read Telegram links: %s", error)
		return {}
	return {str(chat_id): str(customer_id) for chat_id, customer_id in value.items()}


def _save_links(links: dict[str, str]) -> None:
	"""Persist Telegram links atomically without storing customer email addresses."""
	LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = LINKS_PATH.with_suffix(".tmp")
	temporary_path.write_text(json.dumps(links, indent=2), encoding="utf-8")
	temporary_path.replace(LINKS_PATH)


async def link_command(update: Any, context: Any) -> None:
	"""Link the current Telegram chat to a customer by their account email."""
	chat = update.effective_chat
	message = update.effective_message
	if chat is None or message is None:
		return
	if not context.args:
		await message.reply_text("Usage: /link your-account-email@example.com")
		return
	email = context.args[0].strip().lower()
	with SessionLocal() as session:
		customer = session.scalar(
			select(Customer).where(func.lower(Customer.email) == email)
		)
		if customer is None:
			await message.reply_text("I couldn't find a ResolveAI customer with that email.")
			return
		customer_id = customer.id
	links = _load_links()
	links[str(chat.id)] = customer_id
	_save_links(links)
	await message.reply_text(f"This Telegram chat is linked to ResolveAI customer {customer_id}.")


def _post_chat(customer_id: str, chat_id: int, text: str) -> str:
	"""Forward one Telegram message to the ResolveAI HTTP chat endpoint."""
	response = requests.post(
		f"{API_URL}/chat",
		json={
			"conversation_id": f"telegram-{chat_id}",
			"customer_id": customer_id,
			"message": text,
			"channel": "telegram",
		},
		timeout=REQUEST_TIMEOUT_SECONDS,
	)
	response.raise_for_status()
	return str(response.json().get("reply", "I couldn't prepare a reply."))


async def message_handler(update: Any, _: Any) -> None:
	"""Forward linked Telegram text messages to the FastAPI chat endpoint."""
	chat = update.effective_chat
	message = update.effective_message
	if chat is None or message is None or not message.text:
		return
	customer_id = _load_links().get(str(chat.id))
	if customer_id is None:
		await message.reply_text("Link your account first with /link your-account-email@example.com")
		return
	try:
		reply = await asyncio.to_thread(_post_chat, customer_id, chat.id, message.text)
	except requests.RequestException as error:
		logger.exception("Telegram message could not reach ResolveAI")
		await message.reply_text("ResolveAI is temporarily unavailable. Please try again shortly.")
		return
	await message.reply_text(reply)


def build_application() -> Any:
	"""Build the python-telegram-bot application from the configured bot token."""
	from telegram.ext import Application, CommandHandler, MessageHandler, filters

	token = os.getenv("TELEGRAM_BOT_TOKEN")
	if not token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN is required to start the Telegram bot.")
	application = Application.builder().token(token).build()
	application.add_handler(CommandHandler("link", link_command))
	application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
	return application


def main() -> None:
	"""Start polling Telegram updates and forwarding linked customer messages."""
	logging.basicConfig(level=logging.INFO)
	build_application().run_polling()


if __name__ == "__main__":
	main()