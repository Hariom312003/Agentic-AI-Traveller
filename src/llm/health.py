"""
Thread-safe provider health tracking.

REVIEW FIX (#4 — "Global mutable provider-health state with no locking"):
the previous implementation kept a bare module-level ``dict`` that every
request thread read and mutated with unguarded ``+=`` and multi-statement
check-then-act sequences ("if cooldown expired: flip to half-open"). Under
concurrent requests (FastAPI's threadpool for sync code, or several Streamlit
sessions hitting the same backend) that's a textbook lost-update race: two
threads can both read ``consecutive_failures = 2``, both increment to 3,
and only one write survives — the circuit never opens even though the
threshold was crossed twice.

The fix is not "add one lock somewhere" but to make the unsafe pattern
impossible to write: all state lives inside `ProviderHealthRegistry`, every
read-modify-write happens under a single `RLock`, and there is no code path
that touches `_state` from outside this class. `tests/test_llm_router.py`
drives this with real concurrent threads and asserts the counters are exact.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"        # healthy, normal traffic
    OPEN = "open"             # tripped, skip this provider
    HALF_OPEN = "half_open"   # cooldown elapsed, allow one probe request


@dataclass
class ProviderHealth:
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0
    rate_limit_hits: int = 0
    last_success_time: float | None = None
    last_failure_time: float | None = None
    last_error: str | None = None
    circuit_opened_at: float | None = None
    cooldown_until: float | None = None  # explicit provider-given retry-after, if any


class ProviderHealthRegistry:
    """One instance per process, shared by every request. All access goes
    through the lock — no exceptions, no "just this one read is fine"."""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 60.0):
        self._lock = threading.RLock()
        self._state: dict[str, ProviderHealth] = {}
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds

    def _get_or_create(self, provider: str) -> ProviderHealth:
        # Caller must already hold self._lock.
        if provider not in self._state:
            self._state[provider] = ProviderHealth()
        return self._state[provider]

    def record_success(self, provider: str) -> None:
        with self._lock:
            health = self._get_or_create(provider)
            health.consecutive_failures = 0
            health.circuit_state = CircuitState.CLOSED
            health.circuit_opened_at = None
            health.total_successes += 1
            health.last_success_time = time.time()

    def record_failure(self, provider: str, error: str, is_rate_limit: bool = False,
                        retry_after_seconds: float | None = None) -> None:
        with self._lock:
            health = self._get_or_create(provider)
            health.consecutive_failures += 1
            health.total_failures += 1
            health.last_failure_time = time.time()
            health.last_error = error[:500]

            if is_rate_limit:
                health.rate_limit_hits += 1
                cooldown = retry_after_seconds if retry_after_seconds is not None else 5.0
                health.cooldown_until = time.time() + cooldown
                health.circuit_state = CircuitState.OPEN
                health.circuit_opened_at = time.time()
            elif health.consecutive_failures >= self.failure_threshold:
                health.circuit_state = CircuitState.OPEN
                health.circuit_opened_at = time.time()

    def is_available(self, provider: str) -> bool:
        """Returns True if the provider should be attempted right now. Also
        performs the OPEN -> HALF_OPEN transition atomically when the
        cooldown window has elapsed, so callers never have to reason about
        timing themselves."""
        with self._lock:
            health = self._state.get(provider)
            if health is None:
                return True
            if health.circuit_state != CircuitState.OPEN:
                return True

            now = time.time()
            cooldown_end = health.cooldown_until or (
                (health.circuit_opened_at or now) + self.recovery_seconds
            )
            if now >= cooldown_end:
                health.circuit_state = CircuitState.HALF_OPEN
                return True
            return False

    def snapshot(self) -> dict[str, dict]:
        """Point-in-time copy for the monitoring dashboard / API. Returns
        plain dicts (not live references) so the caller can't accidentally
        mutate registry state outside the lock."""
        with self._lock:
            return {name: asdict(health) for name, health in self._state.items()}

    def reset(self) -> None:
        """Test helper — clears all tracked state."""
        with self._lock:
            self._state.clear()


_registry_lock = threading.Lock()
_registry_instance: ProviderHealthRegistry | None = None


def get_health_registry() -> ProviderHealthRegistry:
    """Process-wide singleton, created lazily and exactly once even if
    multiple threads call this during startup."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                from src.config import get_settings

                settings = get_settings()
                _registry_instance = ProviderHealthRegistry(
                    failure_threshold=settings.circuit_failure_threshold,
                    recovery_seconds=settings.circuit_recovery_seconds,
                )
    return _registry_instance
