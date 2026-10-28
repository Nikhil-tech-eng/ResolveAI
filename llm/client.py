"""Google Gemini client for validated JSON model responses."""

import hashlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2                 # attempts per model for transient errors
REQUEST_TIMEOUT_MS = 30000       # per HTTP request (12s was too tight for JSON output)
TOTAL_BUDGET_SECONDS = 75.0      # hard cap per call_llm_json (all retries + models)
PER_MODEL_BUDGET_SECONDS = 35.0  # cap per model so the fallback model always gets a turn
EMBED_BUDGET_SECONDS = 30.0      # hard cap per embed_texts call
COOLDOWN_SECONDS = 20            # applied only after every model fails transiently
INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_PRIMARY_MODEL = "gemini-flash-lite-latest"
DEFAULT_FALLBACK_MODEL = "gemini-flash-lite-latest"

_STATE = {"down_until": 0.0}
DEMO_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_cache.json"
_DEMO_CACHE: dict[str, dict[str, Any]] = {}
_DEMO_CACHE_LOCK = threading.RLock()

_TRANSIENT_CODES = {
    "429", "500", "502", "503", "504",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL",
}
_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
    "DEADLINE_EXCEEDED", "RATE LIMIT", "QUOTA", "TIMEOUT", "TIMED OUT", "OVERLOADED",
)
_TRUE_VALUES = {"1", "true", "yes", "on"}


class LLMClientError(RuntimeError):
    """Raised when Gemini cannot return a response matching the requested schema."""


def _env_flag(name: str, default: str = "false") -> bool:
    """Read a boolean environment flag."""
    return os.getenv(name, default).strip().lower() in _TRUE_VALUES


def _demo_response(agent: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
    """Return a cached response for an agent/input hash, if available."""
    cache_key = _cache_key(agent, user)
    with _DEMO_CACHE_LOCK:
        cached_data = _DEMO_CACHE.get(cache_key)
        if cached_data is None:
            try:
                persisted = json.loads(DEMO_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(persisted, dict):
                    _DEMO_CACHE.update(
                        {
                            str(key): value
                            for key, value in persisted.items()
                            if isinstance(value, dict)
                        }
                    )
            except FileNotFoundError:
                pass
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Could not read demo response cache: %s", error)
            cached_data = _DEMO_CACHE.get(cache_key)
    if cached_data is None:
        return None
    try:
        return schema.model_validate(cached_data)
    except Exception as error:
        logger.warning("Discarding invalid cached %s response: %s", schema.__name__, error)
        with _DEMO_CACHE_LOCK:
            _DEMO_CACHE.pop(cache_key, None)
        return None


def _cache_key(agent: str, user: str) -> str:
    """Hash the owning agent and its exact serialized input."""
    cache_material = "\0".join((agent, user))
    return hashlib.sha256(cache_material.encode("utf-8")).hexdigest()


def _store_demo_response(agent: str, user: str, response: BaseModel) -> None:
    """Store a validated response in memory and the persistent JSON cache."""
    cache_key = _cache_key(agent, user)
    with _DEMO_CACHE_LOCK:
        _DEMO_CACHE[cache_key] = response.model_dump(mode="json")
        try:
            persisted: dict[str, Any] = {}
            if DEMO_CACHE_PATH.exists():
                loaded = json.loads(DEMO_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    persisted = loaded
            persisted[cache_key] = response.model_dump(mode="json")
            DEMO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = DEMO_CACHE_PATH.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(persisted, ensure_ascii=True, indent=2), encoding="utf-8"
            )
            temporary_path.replace(DEMO_CACHE_PATH)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Could not persist demo response cache: %s", error)


def _validate_schema_annotation(annotation: Any, field_name: str) -> None:
    """Reject response schema annotations unsupported by Gemini structured output."""
    if annotation is Any or annotation is dict:
        raise LLMClientError(f"Gemini response field '{field_name}' cannot use Any or dict.")

    origin = get_origin(annotation)
    if origin is dict:
        raise LLMClientError(f"Gemini response field '{field_name}' cannot use dict types.")
    if origin in (Union, UnionType):
        members = get_args(annotation)
        non_optional_members = tuple(member for member in members if member is not type(None))
        if len(members) != 2 or len(non_optional_members) != 1:
            raise LLMClientError(
                f"Gemini response field '{field_name}' cannot use Union types; "
                "use Optional[T] with a None default."
            )
        _validate_schema_annotation(non_optional_members[0], field_name)
        return
    if origin is list:
        members = get_args(annotation)
        if len(members) != 1:
            raise LLMClientError(
                f"Gemini response field '{field_name}' must declare a list item type."
            )
        _validate_schema_annotation(members[0], field_name)
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _validate_response_schema(annotation)


def _validate_response_schema(schema: type[BaseModel]) -> None:
    """Check nested response fields for Gemini-compatible concrete types."""
    for field_name, field in schema.model_fields.items():
        annotation = field.annotation
        origin = get_origin(annotation)
        if (
            origin in (Union, UnionType)
            and type(None) in get_args(annotation)
            and field.default is not None
        ):
            raise LLMClientError(
                f"Optional Gemini response field '{field_name}' must default to None."
            )
        _validate_schema_annotation(annotation, field_name)


def _is_transient_error(error: Exception) -> bool:
    """Identify retryable Gemini errors: quota (429), overload (503), timeouts (504), 5xx."""
    for attribute in ("code", "status_code", "status"):
        value = str(getattr(error, attribute, "")).upper()
        if value in _TRANSIENT_CODES:
            return True
    message = str(error).upper()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _is_timeout_or_503(error: Exception) -> bool:
    """Identify failures that should be retried once on the fallback model."""
    for attribute in ("code", "status_code", "status"):
        if str(getattr(error, attribute, "")).upper() in {"503", "504"}:
            return True
    error_text = f"{type(error).__name__} {error}".upper()
    return isinstance(error, TimeoutError) or any(
        marker in error_text for marker in ("TIMEOUT", "TIMED OUT", "DEADLINE_EXCEEDED", "503")
    )


def _model_chain() -> list[str]:
    """Return the primary model followed by a distinct fallback model."""
    primary = os.getenv("GEMINI_MODEL") or DEFAULT_PRIMARY_MODEL
    fallback = os.getenv("GEMINI_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL
    return [primary] if primary == fallback else [primary, fallback]


def _safe_deterministic_response(schema: type[BaseModel]) -> BaseModel:
    """Build a valid response from declared defaults and empty list fields only."""
    values: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        if not field.is_required():
            values[field_name] = field.get_default(call_default_factory=True)
        elif get_origin(field.annotation) is list:
            values[field_name] = []
        else:
            raise LLMClientError(
                f"No safe deterministic value is defined for required field '{field_name}'."
            )
    try:
        return schema.model_validate(values)
    except Exception as error:
        raise LLMClientError(
            f"Schema defaults do not form a valid deterministic {schema.__name__} response."
        ) from error


def _fallback_response(
    agent: str,
    user: str,
    schema: type[BaseModel],
    demo_mode: bool,
    last_error: Exception,
) -> BaseModel:
    """Return a demo cache hit or a schema-safe deterministic fallback."""
    if demo_mode:
        cached_response = _demo_response(agent, user, schema)
        if cached_response is not None:
            return cached_response
    try:
        fallback = _safe_deterministic_response(schema)
        logger.warning("Using deterministic fallback for %s", schema.__name__)
        return fallback
    except LLMClientError as fallback_error:
        raise LLMClientError(
            f"Gemini failed for {schema.__name__}; no safe response is available: "
            f"{fallback_error}"
        ) from last_error


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter for a zero-based attempt number."""
    return INITIAL_BACKOFF_SECONDS * (2**attempt) + random.random()


def _make_client(api_key: str) -> genai.Client:
    """Create a Gemini client with a per-request timeout and no SDK-level retries."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def call_llm_json(
    system: str,
    user: str,
    schema: type[BaseModel],
    agent: str = "default",
) -> BaseModel:
    """Call Gemini for JSON with retry, model fallback, and a total time budget.

    Transient errors (429/500/502/503/504/timeouts) are retried with backoff on the
    primary model, then on a fallback model. If everything fails, a cached demo
    response or a schema-safe deterministic fallback is returned.
    """
    _validate_response_schema(schema)
    demo_mode = _env_flag("DEMO_MODE")
    if demo_mode:
        cached_response = _demo_response(agent, user, schema)
        if cached_response is not None:
            return cached_response

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_response(
            agent,
            user,
            schema,
            demo_mode,
            LLMClientError("GEMINI_API_KEY is not set; configure it in the .env file."),
        )

    if time.time() < _STATE["down_until"]:
        return _fallback_response(
            agent,
            user,
            schema,
            demo_mode,
            LLMClientError("Gemini skipped after a recent failure."),
        )

    client = _make_client(api_key)
    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
    last_error: Exception = LLMClientError("Gemini did not return a response.")
    all_transient = True

    model_names = _model_chain()
    retry_fallback_once = False
    for model_index, model_name in enumerate(model_names):
        logger.warning("Using Gemini model: %s", model_name)
        model_deadline = min(deadline, time.monotonic() + PER_MODEL_BUDGET_SECONDS)
        attempts = 1 if retry_fallback_once else MAX_ATTEMPTS
        retry_fallback_once = False
        for attempt in range(attempts):
            if time.monotonic() >= model_deadline:
                logger.warning("Time budget exhausted for %s; moving on.", model_name)
                break
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                if response.text is None:
                    raise ValueError("Gemini returned an empty response.")
                validated_response = schema.model_validate_json(response.text)
            except Exception as error:
                last_error = error
                if isinstance(error, ValidationError):
                    # Bad JSON is not an outage; move to the next model without cooldown.
                    all_transient = False
                    logger.warning(
                        "Invalid %s JSON from %s: %s", schema.__name__, model_name, error
                    )
                    break
                if _is_timeout_or_503(error):
                    retry_fallback_once = model_index + 1 < len(model_names)
                    logger.warning(
                        "Gemini call timed out or returned 503 for %s; moving to fallback model.",
                        model_name,
                    )
                    break
                if _is_transient_error(error):
                    backoff = _backoff_seconds(attempt)
                    if attempt + 1 < attempts and time.monotonic() + backoff < model_deadline:
                        logger.warning(
                            "%s transient error (attempt %d/%d), retrying in %.1fs: %r",
                            model_name, attempt + 1, attempts, backoff, error,
                        )
                        time.sleep(backoff)
                        continue
                else:
                    all_transient = False
                logger.warning(
                    "Gemini call failed for %s on %s: %r", schema.__name__, model_name, error
                )
                break  # give up on this model, try the next one

            if demo_mode or _env_flag("DEMO_CACHE_RECORD"):
                _store_demo_response(agent, user, validated_response)
            return validated_response

    # Cool down only when every model failed for transient reasons.
    if all_transient and not isinstance(last_error, ValidationError):
        _STATE["down_until"] = time.time() + COOLDOWN_SECONDS
    return _fallback_response(agent, user, schema, demo_mode, last_error)


def embed_texts(
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """Create Gemini embeddings for KB fallback indexing and query search."""
    if not texts:
        return []
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMClientError("GEMINI_API_KEY is required for Gemini embeddings.")
    client = _make_client(api_key)
    deadline = time.monotonic() + EMBED_BUDGET_SECONDS
    last_error: Exception = LLMClientError("Gemini returned no embeddings.")
    for attempt in range(MAX_ATTEMPTS):
        if time.monotonic() >= deadline:
            break
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(texts):
                raise LLMClientError("Gemini returned an unexpected number of embeddings.")
            return [list(embedding.values) for embedding in embeddings]
        except Exception as error:
            last_error = error
            backoff = _backoff_seconds(attempt)
            if (
                _is_transient_error(error)
                and attempt + 1 < MAX_ATTEMPTS
                and time.monotonic() + backoff < deadline
            ):
                logger.warning(
                    "Gemini embedding transient error (attempt %d/%d), retrying in %.1fs: %r",
                    attempt + 1, MAX_ATTEMPTS, backoff, error,
                )
                time.sleep(backoff)
                continue
            break
    raise LLMClientError("Gemini could not create the requested embeddings.") from last_error