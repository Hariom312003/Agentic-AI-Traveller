"""
Provider abstraction.

Every concrete provider (Gemini, Groq, OpenRouter, OpenAI, Anthropic)
implements this same tiny interface. The router (src/llm/router.py) only
ever talks to `LLMProvider`, never to a specific SDK — that's what makes
adding a 6th provider a one-file change instead of a rewrite.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(Exception):
    """Base class for provider failures. Subclasses let the router make
    retry/failover decisions without string-matching error messages."""


class RateLimitError(ProviderError):
    """429 / quota exhausted — fail over immediately, no local retry."""

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TransientProviderError(ProviderError):
    """5xx / timeout / connection reset — worth a local retry with backoff
    before failing over."""


class NonRetryableProviderError(ProviderError):
    """4xx auth/validation errors etc — retrying won't help, fail over now."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    retry_count: int = 0
    raw: dict[str, Any] | None = field(default=None, repr=False)


class LLMProvider(ABC):
    name: str

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 45.0,
    ) -> str:
        """Return raw text completion. Must raise one of the ProviderError
        subclasses above on failure — never a bare Exception — so the
        router can classify it correctly."""
        raise NotImplementedError

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())
