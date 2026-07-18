"""
LLM router tests, including review issue #4 ("global mutable provider-health
state with no locking"). `test_concurrent_health_updates_are_not_lost` is
the one that actually exercises concurrency: many real OS threads hammer
`record_success`/`record_failure` at once and we assert the final counters
are exactly right. A bare, unlocked dict under this test would
intermittently under-count (lost updates) — this test is deliberately not
using mocked/patched threading, it spins up real `threading.Thread`s.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.base import (
    LLMProvider,
    NonRetryableProviderError,
    RateLimitError,
    TransientProviderError,
)
from src.llm.health import CircuitState, ProviderHealthRegistry
from src.llm.router import AllProvidersExhaustedError, ProviderRouter
from src.config import Settings


class _StubProvider(LLMProvider):
    """Test double: raises a preset sequence of exceptions, then succeeds.
    Lets us script "provider A always rate-limited, provider B works" etc.
    without any network access."""

    name = "stub"
    _behaviors: dict[str, list] = {}  # class-level: provider name -> queue of behaviors

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096, timeout=45.0) -> str:
        queue = self._behaviors.get(self.model, [])
        if queue:
            behavior = queue.pop(0)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior
        return "default stub response"


def _settings_with_providers(tmp_path, priority: str, keys: dict[str, str]) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider_priority=priority,
        gemini_api_key=keys.get("gemini"), groq_api_key=keys.get("groq"),
        openrouter_api_key=keys.get("openrouter"), openai_api_key=keys.get("openai"),
        anthropic_api_key=keys.get("anthropic"),
        chroma_db_path=str(tmp_path / "chroma"), memory_db_path=str(tmp_path / "mem.db"),
        checkpoint_db_path=str(tmp_path / "cp.sqlite"),
        circuit_failure_threshold=3, circuit_recovery_seconds=60.0, llm_max_retries_per_provider=1,
    )


@pytest.fixture()
def patched_router(monkeypatch, tmp_path):
    """Router wired to `_StubProvider` for every provider name, so tests
    control exactly what each "provider" does without touching a real SDK."""
    from src.llm import router as router_module

    monkeypatch.setattr(router_module, "PROVIDER_CLASSES", {
        name: _StubProvider for name in ["gemini", "groq", "openrouter", "openai", "anthropic", "gemini_backup", "groq_backup"]
    })
    _StubProvider._behaviors = {}
    settings = _settings_with_providers(
        tmp_path, "gemini,groq,openrouter",
        {"gemini": "k1", "groq": "k2", "openrouter": "k3"},
    )
    registry = ProviderHealthRegistry(failure_threshold=3, recovery_seconds=60.0)
    return ProviderRouter(settings=settings, health_registry=registry)


def test_uses_first_healthy_provider(patched_router):
    _StubProvider._behaviors = {"gemini-2.5-flash": ["hello from gemini"]}
    response = patched_router.generate("hi")
    assert response.provider == "gemini"
    assert response.text == "hello from gemini"


def test_fails_over_to_next_provider_on_rate_limit(patched_router):
    _StubProvider._behaviors = {
        patched_router.settings.gemini_model: [RateLimitError("429 quota exceeded")],
        patched_router.settings.groq_model: ["hello from groq"],
    }
    response = patched_router.generate("hi")
    assert response.provider == "groq"
    assert response.text == "hello from groq"


def test_rate_limit_opens_circuit_immediately(patched_router):
    _StubProvider._behaviors = {
        patched_router.settings.gemini_model: [RateLimitError("429")],
        patched_router.settings.groq_model: ["ok"],
    }
    patched_router.generate("hi")
    assert patched_router.health.is_available("gemini") is False
    snapshot = patched_router.health.snapshot()
    assert snapshot["gemini"]["circuit_state"] == CircuitState.OPEN


def test_transient_error_retries_locally_before_failing_over(patched_router):
    _StubProvider._behaviors = {
        patched_router.settings.gemini_model: [TransientProviderError("503"), "recovered on retry"],
    }
    response = patched_router.generate("hi")
    assert response.provider == "gemini"
    assert response.retry_count == 1


def test_non_retryable_error_fails_over_without_local_retry(patched_router):
    _StubProvider._behaviors = {
        patched_router.settings.gemini_model: [NonRetryableProviderError("400 bad request")],
        patched_router.settings.groq_model: ["ok from groq"],
    }
    response = patched_router.generate("hi")
    assert response.provider == "groq"


def test_all_providers_exhausted_raises_with_attempt_detail(patched_router):
    _StubProvider._behaviors = {
        patched_router.settings.gemini_model: [TransientProviderError("down")] * 5,
        patched_router.settings.groq_model: [TransientProviderError("down")] * 5,
        patched_router.settings.openrouter_model: [TransientProviderError("down")] * 5,
    }
    with pytest.raises(AllProvidersExhaustedError) as exc_info:
        patched_router.generate("hi")
    assert len(exc_info.value.attempts) == 3


def test_no_configured_providers_raises_immediately(tmp_path):
    settings = _settings_with_providers(tmp_path, "gemini,groq", {})
    router = ProviderRouter(settings=settings, health_registry=ProviderHealthRegistry())
    with pytest.raises(AllProvidersExhaustedError):
        router.generate("hi")


def test_circuit_transitions_to_half_open_after_recovery_window():
    registry = ProviderHealthRegistry(failure_threshold=2, recovery_seconds=0.05)
    registry.record_failure("gemini", "err1")
    registry.record_failure("gemini", "err2")
    assert registry.is_available("gemini") is False
    time.sleep(0.06)
    assert registry.is_available("gemini") is True  # transitions OPEN -> HALF_OPEN
    snapshot = registry.snapshot()
    assert snapshot["gemini"]["circuit_state"] == CircuitState.HALF_OPEN


# ---- REVIEW FIX #4: thread safety ------------------------------------------

def test_concurrent_health_updates_are_not_lost():
    """The regression test for 'global mutable provider-health state with no
    locking'. 50 threads each record exactly one failure and one success;
    with correct locking the totals must be exactly 50/50 no matter how
    the OS interleaves them. This is deliberately real threads, not a mock,
    because the bug was a genuine race condition, not a logic error a
    single-threaded test could ever catch."""
    registry = ProviderHealthRegistry(failure_threshold=1_000_000, recovery_seconds=60.0)
    n_threads = 50
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()  # maximize actual concurrent contention on the lock
        registry.record_failure("gemini", "synthetic failure")
        registry.record_success("gemini")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    snapshot = registry.snapshot()["gemini"]
    assert snapshot["total_failures"] == n_threads, "lost updates under concurrency (missing lock)"
    assert snapshot["total_successes"] == n_threads, "lost updates under concurrency (missing lock)"


def test_concurrent_mixed_providers_do_not_cross_contaminate():
    registry = ProviderHealthRegistry(failure_threshold=1_000_000, recovery_seconds=60.0)
    n_per_provider = 30
    providers = ["gemini", "groq", "openrouter"]
    barrier = threading.Barrier(n_per_provider * len(providers))

    def worker(provider_name: str):
        barrier.wait()
        registry.record_failure(provider_name, "err")

    threads = [threading.Thread(target=worker, args=(p,)) for p in providers for _ in range(n_per_provider)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    snapshot = registry.snapshot()
    for p in providers:
        assert snapshot[p]["total_failures"] == n_per_provider
