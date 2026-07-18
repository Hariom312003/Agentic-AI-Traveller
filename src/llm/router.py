"""
Multi-provider LLM router.

Responsibilities, in order, for every `generate()` call:
1. Walk `settings.provider_priority_list()`, skipping providers with no key
   configured and providers whose circuit is currently OPEN
   (`ProviderHealthRegistry.is_available`).
2. For the provider being attempted, retry transient errors in-process with
   exponential backoff (via `tenacity`); fail over immediately (no local
   retry) on rate limits or non-retryable errors — retrying a 429 against
   the same key just burns more quota for no benefit.
3. Record every attempt (success or failure) in the health registry and in
   `AgentExecutionRecord`-compatible telemetry.
4. If every provider is exhausted, raise `AllProvidersExhaustedError` — the
   caller (an agent) decides whether that means "use the rule-based
   fallback" or "surface an error", the router doesn't assume.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings
from src.llm.base import (
    LLMResponse,
    NonRetryableProviderError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from src.llm.health import ProviderHealthRegistry, get_health_registry
from src.llm.providers import PROVIDER_CLASSES
from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)


class AllProvidersExhaustedError(Exception):
    def __init__(self, attempts: list["ProviderAttempt"]):
        self.attempts = attempts
        summary = "; ".join(f"{a.provider}: {a.error}" for a in attempts) or "no providers configured"
        super().__init__(f"All LLM providers exhausted. {summary}")


@dataclass
class ProviderAttempt:
    provider: str
    model: str
    success: bool
    latency_ms: float
    retries: int = 0
    error: str | None = None


class ProviderRouter:
    def __init__(self, settings: Settings | None = None, health_registry: ProviderHealthRegistry | None = None):
        self.settings = settings or get_settings()
        self.health = health_registry or get_health_registry()

    # -- provider selection -------------------------------------------------
    def configured_providers(self) -> list[str]:
        return [p for p in self.settings.provider_priority_list() if self.settings.key_for(p)]

    def _build_provider(self, name: str):
        key = self.settings.key_for(name)
        model = self.settings.model_for(name)
        cls = PROVIDER_CLASSES.get(name)
        if cls is None or not key:
            return None
        return cls(api_key=key, model=model)

    # -- single-provider call with local retry ------------------------------
    def _call_with_retry(self, provider_name: str, provider, prompt: str, system_prompt: str | None,
                          temperature: float, max_tokens: int) -> tuple[str, int]:
        attempt_counter = {"n": 0}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.llm_max_retries_per_provider + 1),
            wait=wait_exponential(multiplier=1.5, min=1, max=20),
            retry=retry_if_exception_type(TransientProviderError),
        )
        def _do_call() -> str:
            attempt_counter["n"] += 1
            return provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.settings.llm_timeout_seconds,
            )

        text = _do_call()
        return text, attempt_counter["n"] - 1  # retries = attempts beyond the first

    # -- public entrypoint ----------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        candidates = self.configured_providers()
        attempts: list[ProviderAttempt] = []

        if not candidates:
            raise AllProvidersExhaustedError(attempts)

        for provider_name in candidates:
            if not self.health.is_available(provider_name):
                logger.info("skip_provider_circuit_open", extra={"provider": provider_name})
                continue

            provider = self._build_provider(provider_name)
            if provider is None:
                continue

            start = time.perf_counter()
            try:
                text, retries = self._call_with_retry(
                    provider_name, provider, prompt, system_prompt, temperature, max_tokens
                )
                latency_ms = (time.perf_counter() - start) * 1000
                self.health.record_success(provider_name)
                attempts.append(ProviderAttempt(provider_name, provider.model, True, latency_ms, retries))
                logger.info(
                    "llm_call_success",
                    extra={"provider": provider_name, "model": provider.model, "latency_ms": round(latency_ms, 1),
                           "retries": retries},
                )
                return LLMResponse(
                    text=text, provider=provider_name, model=provider.model,
                    latency_ms=latency_ms, retry_count=retries,
                )

            except RateLimitError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                self.health.record_failure(
                    provider_name, str(exc), is_rate_limit=True,
                    retry_after_seconds=exc.retry_after_seconds,
                )
                attempts.append(ProviderAttempt(provider_name, provider.model, False, latency_ms, error=str(exc)))
                logger.warning("llm_rate_limited_failover", extra={"provider": provider_name, "error": str(exc)})
                continue

            except (TransientProviderError, NonRetryableProviderError, ProviderError) as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                self.health.record_failure(provider_name, str(exc))
                attempts.append(ProviderAttempt(provider_name, provider.model, False, latency_ms, error=str(exc)))
                logger.warning("llm_call_failed_failover", extra={"provider": provider_name, "error": str(exc)})
                continue

            except Exception as exc:  # noqa: BLE001 - last line of defense, still fails over
                latency_ms = (time.perf_counter() - start) * 1000
                self.health.record_failure(provider_name, str(exc))
                attempts.append(ProviderAttempt(provider_name, provider.model, False, latency_ms, error=str(exc)))
                logger.error("llm_call_unexpected_error", extra={"provider": provider_name, "error": str(exc)})
                continue

        raise AllProvidersExhaustedError(attempts)


_router_singleton: ProviderRouter | None = None


def get_router() -> ProviderRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = ProviderRouter()
    return _router_singleton
