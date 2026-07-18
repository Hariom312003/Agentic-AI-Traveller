"""
Shared agent infrastructure.

`generate_structured` is the one function every LLM-calling agent uses to
get JSON back from a model: call the router, extract JSON from the raw
text (models routinely wrap it in prose or markdown fences despite
instructions not to), validate it against a pydantic model, and on failure
retry with the validation error fed back into the prompt — this is the
"output schema verification" the code review called out as a strength, now
centralized instead of re-implemented per agent.

It never silently swallows a final failure: after `max_retries` it raises
`SchemaGenerationError`, and the calling agent is responsible for deciding
whether to fall back to a rule-based path (most do — see
src/planning_engine/fallback_planner.py) or propagate the error.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.config import get_settings
from src.llm.router import AllProvidersExhaustedError, get_router
from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class SchemaGenerationError(Exception):
    def __init__(self, message: str, last_raw_response: str | None = None):
        super().__init__(message)
        self.last_raw_response = last_raw_response


def extract_json(raw_text: str) -> dict | list:
    """Models frequently wrap JSON in ```json fences or add a sentence
    before/after despite being told not to. Try, in order: the whole
    string as-is, the first fenced code block, then the widest {...} or
    [...] span in the text."""
    candidates = [raw_text.strip()]

    fence_match = _JSON_BLOCK_RE.search(raw_text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    obj_start, obj_end = raw_text.find("{"), raw_text.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(raw_text[obj_start : obj_end + 1])

    arr_start, arr_end = raw_text.find("["), raw_text.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(raw_text[arr_start : arr_end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise ValueError(f"Could not extract valid JSON from model response: {last_error}")


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    temperature: float = 0.3,
    max_retries: int | None = None,
) -> tuple[T, "LLMCallMeta"]:
    """Call the LLM router, parse + validate the response against `schema`,
    retrying with the validation error appended to the prompt on failure.
    Returns (validated_instance, call_metadata) so agents can attach
    provider/model/retry info to their telemetry record."""
    settings = get_settings()
    router = get_router()
    max_retries = settings.max_json_parse_retries if max_retries is None else max_retries

    schema_hint = (
        f"\n\nRespond with ONLY valid JSON matching this shape (no prose, no markdown fences):\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )
    prompt = user_prompt + schema_hint
    last_raw: str | None = None
    last_error: Exception | None = None
    total_retries = 0
    provider_used = model_used = None

    for attempt in range(max_retries + 1):
        response = router.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
        last_raw = response.text
        provider_used, model_used = response.provider, response.model
        total_retries += response.retry_count
        try:
            parsed = extract_json(response.text)
            instance = schema.model_validate(parsed)
            return instance, LLMCallMeta(provider_used, model_used, total_retries + attempt)
        except (ValueError, ValidationError) as exc:
            last_error = exc
            logger.warning("schema_validation_retry", extra={"attempt": attempt, "error": str(exc)[:300]})
            prompt = (
                f"{user_prompt}{schema_hint}\n\n"
                f"Your previous response failed validation with this error:\n{exc}\n"
                f"Previous response was:\n{response.text[:1500]}\n\n"
                f"Fix it and return ONLY the corrected JSON."
            )

    raise SchemaGenerationError(f"Schema validation failed after {max_retries + 1} attempts: {last_error}", last_raw)


class LLMCallMeta:
    def __init__(self, provider: str | None, model: str | None, retries: int):
        self.provider = provider
        self.model = model
        self.retries = retries


__all__ = ["generate_structured", "extract_json", "SchemaGenerationError", "AllProvidersExhaustedError"]
