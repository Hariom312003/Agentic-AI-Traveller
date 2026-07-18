"""
Central configuration for AI Traveller.

Everything that can vary between a laptop, a CI runner, and a production
container lives here, loaded once as a singleton (`get_settings()`), typed
and validated by pydantic so a bad `.env` fails fast at startup instead of
half way through a graph run.

Design choice: no module ever reads `os.environ` directly outside this file.
That's what let us fix "hardcoded model strings" and "silently guessed API
keys" as a category, not a one-off patch.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ProviderName = Literal[
    "gemini", "gemini_backup", "groq", "groq_backup", "openrouter", "openai", "anthropic"
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM provider keys (all optional — router only uses what's set) ----
    gemini_api_key: str | None = None
    gemini_api_key_backup: str | None = None
    groq_api_key: str | None = None
    groq_api_key_backup: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    llm_provider_priority: str = (
        "gemini,groq,openrouter,gemini_backup,groq_backup,openai,anthropic"
    )

    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-5"

    embedding_provider: Literal["gemini", "openai", "local"] = "gemini"
    embedding_model: str = "text-embedding-004"
    embedding_dimensions: int = 384  # used by the local fallback embedder

    # ---- Resilience tuning ----
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 60.0
    llm_max_retries_per_provider: int = 2
    llm_timeout_seconds: float = 45.0

    # ---- Storage ----
    chroma_db_path: str = "./chroma_db"
    memory_db_path: str = "./data/memory.db"
    checkpoint_db_path: str = "./data/checkpoints.sqlite"
    destinations_data_path: str = "./data/destinations"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:8501,http://localhost:8000"

    # ---- Misc ----
    log_level: str = "INFO"
    environment: Literal["development", "test", "production"] = "development"
    anonymized_telemetry: bool = False
    default_currency: str = "INR"

    # ---- Validation thresholds (kept here, not scattered as magic numbers) --
    duplicate_fuzzy_threshold: float = 80.0
    max_planner_repair_attempts: int = 2
    max_json_parse_retries: int = 2

    @field_validator("llm_provider_priority")
    @classmethod
    def _normalize_priority(cls, v: str) -> str:
        return ",".join(p.strip().lower() for p in v.split(",") if p.strip())

    # ---- Derived helpers ----
    def provider_priority_list(self) -> list[str]:
        return [p for p in self.llm_provider_priority.split(",") if p]

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def key_for(self, provider: str) -> str | None:
        return {
            "gemini": self.gemini_api_key,
            "gemini_backup": self.gemini_api_key_backup,
            "groq": self.groq_api_key,
            "groq_backup": self.groq_api_key_backup,
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(provider)

    def model_for(self, provider: str) -> str:
        return {
            "gemini": self.gemini_model,
            "gemini_backup": self.gemini_model,
            "groq": self.groq_model,
            "groq_backup": self.groq_model,
            "openrouter": self.openrouter_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
        }.get(provider, "unknown")

    def ensure_directories(self) -> None:
        for path_str in (self.chroma_db_path, self.memory_db_path, self.checkpoint_db_path):
            p = Path(path_str)
            (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. `lru_cache` makes repeated calls free."""
    settings = Settings()
    settings.ensure_directories()
    return settings
