"""
Concrete provider implementations.

Each class wraps one vendor SDK and normalizes its exceptions into the
`ProviderError` hierarchy from `base.py`. This is the only file that is
allowed to import a vendor SDK — everything above it (router, agents)
only ever sees `LLMProvider.generate()` and `ProviderError` subclasses.

All 5 SDKs used here (google-genai, groq, openai, anthropic) are the
current, actively maintained clients for their respective vendors,
version-pinned in requirements.txt. Signatures were verified against the
installed packages while building this project, not assumed from memory.
"""
from __future__ import annotations

import time

from src.llm.base import (
    LLMProvider,
    NonRetryableProviderError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)

_TRANSIENT_CODES = {500, 502, 503, 504}


def _classify_http_error(status_code: int | None, message: str, retry_after: float | None = None) -> ProviderError:
    if status_code == 429:
        return RateLimitError(message, retry_after_seconds=retry_after)
    if status_code in _TRANSIENT_CODES or status_code is None:
        # None => connection/timeout error, also worth a local retry
        return TransientProviderError(message)
    return NonRetryableProviderError(message)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096, timeout=45.0) -> str:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types

        client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        try:
            response = client.models.generate_content(model=self.model, contents=prompt, config=config)
        except genai_errors.APIError as exc:
            raise _classify_http_error(getattr(exc, "code", None), str(exc)) from exc
        except TimeoutError as exc:
            raise TransientProviderError(str(exc)) from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise TransientProviderError("Gemini returned an empty response")
        return text.strip()


class GroqProvider(LLMProvider):
    name = "groq"

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096, timeout=45.0) -> str:
        from groq import Groq
        import groq as groq_sdk

        client = Groq(api_key=self.api_key, timeout=timeout)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except groq_sdk.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except (groq_sdk.APIConnectionError, groq_sdk.APITimeoutError, groq_sdk.InternalServerError) as exc:
            raise TransientProviderError(str(exc)) from exc
        except groq_sdk.APIStatusError as exc:
            raise _classify_http_error(exc.status_code, str(exc)) from exc

        text = completion.choices[0].message.content
        if not text or not text.strip():
            raise TransientProviderError("Groq returned an empty response")
        return text.strip()


class OpenRouterProvider(LLMProvider):
    """OpenRouter speaks the OpenAI-compatible chat completions schema, so we
    reuse the `openai` SDK pointed at a different base_url rather than
    hand-rolling HTTP calls."""

    name = "openrouter"

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096, timeout=45.0) -> str:
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            default_headers={"HTTP-Referer": "https://github.com/ai-traveller", "X-Title": "AI Traveller"},
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            raise TransientProviderError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise _classify_http_error(exc.status_code, str(exc)) from exc

        text = completion.choices[0].message.content
        if not text or not text.strip():
            raise TransientProviderError("OpenRouter returned an empty response")
        return text.strip()


class OpenAIProvider(LLMProvider):
    name = "openai"

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096, timeout=45.0) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key, timeout=timeout)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            raise TransientProviderError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise _classify_http_error(exc.status_code, str(exc)) from exc

        text = completion.choices[0].message.content
        if not text or not text.strip():
            raise TransientProviderError("OpenAI returned an empty response")
        return text.strip()


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096, timeout=45.0) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout)
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            message = client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError,
                anthropic.OverloadedError) as exc:
            raise TransientProviderError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise _classify_http_error(exc.status_code, str(exc)) from exc

        blocks = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        text = "".join(blocks)
        if not text or not text.strip():
            raise TransientProviderError("Anthropic returned an empty response")
        return text.strip()


PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "gemini_backup": GeminiProvider,
    "groq": GroqProvider,
    "groq_backup": GroqProvider,
    "openrouter": OpenRouterProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}
