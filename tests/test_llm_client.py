"""Mocked tests for Gemini retries, schema validation, and safe fallback."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from llm import client as llm_client


class AnswerSchema(BaseModel):
	"""Small Gemini-compatible response schema used by mocked tests."""

	answer: str


class FallbackSchema(BaseModel):
	"""Response schema with deterministic safe defaults."""

	decision: str = "escalate"
	items: list[str] = []
	note: str | None = None


class QuotaError(Exception):
	"""Mock Gemini quota exception."""

	code = 429


@pytest.fixture(autouse=True)
def configure_mock_environment(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	"""Configure credentials and clear the demo cache for each test."""
	monkeypatch.setenv("GEMINI_API_KEY", "mock-api-key")
	monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
	monkeypatch.setenv("DEMO_MODE", "false")
	monkeypatch.setattr(llm_client, "DEMO_CACHE_PATH", tmp_path / "demo_cache.json")
	llm_client._DEMO_CACHE.clear()


def mock_gemini(monkeypatch: pytest.MonkeyPatch, responses: list[object]) -> Mock:
	"""Install a mocked Google GenAI client with queued responses or exceptions."""
	client = Mock()
	client.models.generate_content.side_effect = responses
	monkeypatch.setattr(llm_client.genai, "Client", Mock(return_value=client))
	return client


def test_success_uses_structured_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Validate the response schema and JSON configuration passed to Gemini."""
	client = mock_gemini(
		monkeypatch,
		[SimpleNamespace(text=json.dumps({"answer": "verified"}))],
	)
	result = llm_client.call_llm_json("system", "user", AnswerSchema)
	call = client.models.generate_content.call_args
	assert result == AnswerSchema(answer="verified")
	assert call.kwargs["config"].response_mime_type == "application/json"
	assert call.kwargs["config"].response_schema is AnswerSchema
	assert client.models.generate_content.call_count == 1


def test_quota_error_retries_with_exponential_backoff(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Retry quota errors and stop after the first successful generation."""
	client = mock_gemini(
		monkeypatch,
		[
			QuotaError("quota exceeded"),
			QuotaError("quota exceeded"),
			SimpleNamespace(text=json.dumps({"answer": "recovered"})),
		],
	)
	sleep = Mock()
	monkeypatch.setattr(llm_client.time, "sleep", sleep)

	result = llm_client.call_llm_json("system", "user", AnswerSchema)

	assert result == AnswerSchema(answer="recovered")
	assert client.models.generate_content.call_count == 3
	assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]


def test_quota_exhaustion_uses_safe_schema_defaults(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Use safe defaults after three quota failures without further model calls."""
	client = mock_gemini(
		monkeypatch,
		[QuotaError("429 quota"), QuotaError("429 quota"), QuotaError("429 quota")],
	)
	monkeypatch.setattr(llm_client.time, "sleep", Mock())

	result = llm_client.call_llm_json("system", "user", FallbackSchema)

	assert result == FallbackSchema()
	assert client.models.generate_content.call_count == 3


def test_demo_mode_cache_is_used_after_quota_exhaustion(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Fall back to an exact-prompt cached response after quota retries fail."""
	monkeypatch.setenv("DEMO_MODE", "true")
	client = mock_gemini(
		monkeypatch,
		[QuotaError("429 quota"), QuotaError("429 quota"), QuotaError("429 quota")],
	)
	cached = AnswerSchema(answer="cached demo answer")
	cache_lookup = Mock(side_effect=[None, cached])
	monkeypatch.setattr(llm_client, "_demo_response", cache_lookup)
	monkeypatch.setattr(llm_client.time, "sleep", Mock())

	result = llm_client.call_llm_json("system", "user", AnswerSchema)

	assert result == cached
	assert client.models.generate_content.call_count == 3
	assert cache_lookup.call_count == 2


def test_invalid_json_does_not_make_a_second_generation_call(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Avoid an extra LLM call for invalid output and use a safe fallback."""
	client = mock_gemini(monkeypatch, [SimpleNamespace(text="not json")])

	result = llm_client.call_llm_json("system", "user", FallbackSchema)

	assert result == FallbackSchema()
	assert client.models.generate_content.call_count == 1


def test_invalid_response_without_safe_defaults_raises_clear_error(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Raise a clear client error when a required field has no safe value."""
	client = mock_gemini(monkeypatch, [SimpleNamespace(text="not json")])
	with pytest.raises(llm_client.LLMClientError, match="no safe response is available"):
		llm_client.call_llm_json("system", "user", AnswerSchema)
	assert client.models.generate_content.call_count == 1


def test_non_quota_api_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Use a safe fallback without retrying non-quota API errors."""
	client = mock_gemini(monkeypatch, [RuntimeError("service unavailable")])
	result = llm_client.call_llm_json("system", "user", FallbackSchema)
	assert result == FallbackSchema()
	assert client.models.generate_content.call_count == 1


def test_incompatible_dict_schema_is_rejected_before_api_call(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Reject generic dictionary fields before contacting Gemini."""
	from typing import Any

	class BadSchema(BaseModel):
		payload: dict[str, Any]

	client = mock_gemini(monkeypatch, [])
	with pytest.raises(llm_client.LLMClientError, match="cannot use dict types"):
		llm_client.call_llm_json("system", "user", BadSchema)
	client.models.generate_content.assert_not_called()


def test_union_response_field_is_rejected_before_api_call(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Reject non-optional union response fields before contacting Gemini."""
	class BadUnionSchema(BaseModel):
		value: str | int

	client = mock_gemini(monkeypatch, [])
	with pytest.raises(llm_client.LLMClientError, match="cannot use Union types"):
		llm_client.call_llm_json("system", "user", BadUnionSchema)
	client.models.generate_content.assert_not_called()


def test_demo_mode_caches_successful_validated_response(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Reuse a successful response for an exact prompt while demo mode is enabled."""
	monkeypatch.setenv("DEMO_MODE", "true")
	client = mock_gemini(
		monkeypatch,
		[SimpleNamespace(text=json.dumps({"answer": "cached"}))],
	)
	first = llm_client.call_llm_json("system", "user", AnswerSchema)
	second = llm_client.call_llm_json("system", "user", AnswerSchema)

	assert first == second == AnswerSchema(answer="cached")
	assert client.models.generate_content.call_count == 1


def test_demo_cache_persists_and_uses_agent_input_hash(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Reload a recorded response from disk under its agent/input cache key."""
	monkeypatch.setenv("DEMO_MODE", "true")
	first_client = mock_gemini(
		monkeypatch,
		[SimpleNamespace(text=json.dumps({"answer": "recorded"}))],
	)
	first = llm_client.call_llm_json("system", "input", AnswerSchema, "intent")
	key = llm_client._cache_key("intent", "input")
	persisted = json.loads(llm_client.DEMO_CACHE_PATH.read_text(encoding="utf-8"))
	assert key in persisted
	assert llm_client._cache_key("context", "input") != key

	llm_client._DEMO_CACHE.clear()
	second_client = mock_gemini(monkeypatch, [])
	second = llm_client.call_llm_json("changed system", "input", AnswerSchema, "intent")

	assert first == second == AnswerSchema(answer="recorded")
	assert first_client.models.generate_content.call_count == 1
	second_client.models.generate_content.assert_not_called()