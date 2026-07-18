"""
Pluggable embeddings.

Two implementations behind one interface:

- `GeminiEmbeddingProvider` — production quality, calls the Gemini
  embedding API. Requires `GEMINI_API_KEY` and network access.
- `LocalHashingEmbeddingProvider` — deterministic, fully offline, zero
  downloads. Uses scikit-learn's `HashingVectorizer` (word n-grams hashed
  into a fixed number of dimensions) as a real, if lower-quality, dense
  representation: texts sharing vocabulary land closer in the hashed space
  even though there's no learned semantics. This is what lets the whole RAG
  pipeline run and be unit-tested in an environment with no access to an
  embedding API (or no key configured yet) — see tests/test_rag.py, which
  runs entirely on this provider.

`get_embedding_provider()` picks based on `settings.embedding_provider`,
and — importantly — falls back to the local provider automatically if the
configured remote one isn't actually usable (no key), rather than crashing
the whole pipeline at import time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str | None = None, dimensions: int = 768):
        self.api_key = api_key
        self._configured_model = model
        self.model = None  # Resolved dynamically
        self.dimensions = dimensions

    def _resolve_model(self) -> str:
        if self.model is not None:
            return self.model
        from google import genai
        client = genai.Client(api_key=self.api_key)
        try:
            available_models = [m.name for m in client.models.list()]
            available_models_clean = [m.replace("models/", "") for m in available_models]
            preferred = ["gemini-embedding-2", "text-embedding-004", "gemini-embedding-001"]

            if self._configured_model:
                conf_model_clean = self._configured_model.replace("models/", "")
                if conf_model_clean in available_models_clean:
                    self.model = self._configured_model
                    return self.model

            for p in preferred:
                if p in available_models_clean:
                    idx = available_models_clean.index(p)
                    self.model = available_models[idx]
                    return self.model

            for idx, m_clean in enumerate(available_models_clean):
                if "embedding" in m_clean:
                    self.model = available_models[idx]
                    return self.model

            self.model = self._configured_model or "text-embedding-004"
        except Exception as exc:
            logger.warning(f"Error listing Gemini embedding models: {exc}. Using default fallback name.")
            self.model = self._configured_model or "text-embedding-004"
        return self.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._resolve_model()
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        result = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        return [list(e.values) for e in result.embeddings]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        kwargs = {"input": texts, "model": self.model}
        if self.model in ("text-embedding-3-small", "text-embedding-3-large"):
            kwargs["dimensions"] = self.dimensions
        response = client.embeddings.create(**kwargs)
        return [data.embedding for data in response.data]


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline fallback. Not a semantic embedding model — a
    hashed bag-of-word-ngrams projected into a fixed number of dimensions
    and L2-normalized. Two texts sharing vocabulary get non-trivial cosine
    similarity; texts sharing nothing land near-orthogonal. That's enough
    signal to make hybrid retrieval (this + BM25) meaningfully testable
    without any network access or model download.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=dimensions, alternate_sign=False, norm="l2", ngram_range=(1, 2), analyzer="word"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        return dense.tolist()


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Transparent proxy that delegates to a primary embedding provider (e.g. Gemini),
    and on any exception (timeout, 404, 429, auth), gracefully falls back to the
    offline LocalHashingEmbeddingProvider with matching dimensions so that RAG
    operations never crash the app.
    """

    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider):
        self.primary = primary
        self.fallback = fallback
        self.dimensions = primary.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed(texts)
        except Exception as exc:
            logger.warning(
                f"Primary embedding provider '{getattr(self.primary, 'model', 'unknown')}' "
                f"failed with: {exc}. Falling back to local hashing."
            )
            return self.fallback.embed(texts)


def get_embedding_provider() -> EmbeddingProvider:
    from src.config import get_settings

    settings = get_settings()
    local_provider = LocalHashingEmbeddingProvider(dimensions=settings.embedding_dimensions)

    if settings.embedding_provider == "gemini" and settings.gemini_api_key:
        gemini_provider = GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        return FallbackEmbeddingProvider(primary=gemini_provider, fallback=local_provider)

    elif settings.embedding_provider == "openai" and settings.openai_api_key:
        openai_provider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model or "text-embedding-3-small",
            dimensions=settings.embedding_dimensions,
        )
        return FallbackEmbeddingProvider(primary=openai_provider, fallback=local_provider)

    return local_provider
