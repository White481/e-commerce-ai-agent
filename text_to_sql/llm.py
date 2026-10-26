"""LLM clients. The agent only needs "system + user prompt in, text out", so any
provider fits behind LLMClient and tests can pass a fake one.

Every HTTP call is recorded (model, time, error) so slow answers can be traced
to retries and fallbacks instead of guessed at.
"""

import time
from dataclasses import dataclass, field
from typing import Protocol

from text_to_sql.config import Settings

_OVERLOADED_CODES = {429, 500, 503, 504}


@dataclass
class LLMCall:
    model: str
    ms: int
    error: str | None = None


@dataclass
class LLMReply:
    text: str
    model: str
    calls: list[LLMCall] = field(default_factory=list)


class LLMError(Exception):
    """The LLM call failed (network, quota, bad key). Not something a retry prompt fixes."""

    def __init__(
        self,
        message: str,
        overloaded: bool = False,
        calls: list[LLMCall] | None = None,
        retry_same_model: bool = True,
    ):
        super().__init__(message)
        # Busy or rate-limited (429/5xx/timeout): another model may still answer
        self.overloaded = overloaded
        # False for a daily quota: it won't come back in seconds, but other models have their own
        self.retry_same_model = retry_same_model
        self.calls = calls or []


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> LLMReply: ...


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_ms: int = 30_000,
        thinking_level: str = "low",
        max_tries: int = 3,
        backoff_s: float = 1.0,
    ):
        from google import genai  # imported here so tests without the key don't need it
        from google.genai import types

        # Retries are done here, not by the SDK, so each one shows up in the log.
        # The timeout is per request, so one stuck call can't hold /ask for minutes.
        options = types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1))
        self._client = genai.Client(api_key=api_key, http_options=options)
        self.model = model
        self.timeout_ms = timeout_ms
        self.max_tries = max_tries
        self.backoff_s = backoff_s
        # Gemini 3 thinks by default, which took minutes per question; SQL over 4 tables needs little
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level.upper()) if thinking_level else None

    def generate(self, system: str, user: str) -> LLMReply:
        calls: list[LLMCall] = []
        for attempt in range(self.max_tries):
            if attempt:
                time.sleep(min(self.backoff_s * 2 ** (attempt - 1), 4.0))
            started = time.monotonic()
            try:
                text = self._call(system, user)
            except LLMError as e:
                calls.append(LLMCall(self.model, _ms_since(started), str(e)))
                if not (e.overloaded and e.retry_same_model):
                    e.calls = calls
                    raise
                last = e
                continue
            calls.append(LLMCall(self.model, _ms_since(started)))
            return LLMReply(text, self.model, calls)
        last.calls = calls
        raise last

    def _call(self, system: str, user: str) -> str:
        import httpx
        from google.genai import errors, types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                # temperature 0: the same question should give the same SQL
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0, thinking_config=self._thinking
                ),
            )
        except errors.APIError as e:
            daily_quota = e.code == 429 and "PerDay" in str(e.details)
            raise LLMError(
                f"LLM request failed ({e.code} {e.status}): {e.message}",
                overloaded=e.code in _OVERLOADED_CODES,
                retry_same_model=not daily_quota,
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(f"LLM request timed out after {self.timeout_ms / 1000:.0f}s", overloaded=True) from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {type(e).__name__}") from e
        return response.text or ""


class FallbackLLM:
    """Try each model in order, moving on only when one is overloaded."""

    def __init__(self, clients: list[LLMClient]):
        self.clients = clients

    def generate(self, system: str, user: str) -> LLMReply:
        calls: list[LLMCall] = []
        errors = []
        for client in self.clients:
            try:
                reply = client.generate(system, user)
            except LLMError as e:
                calls += e.calls
                if not e.overloaded:
                    e.calls = calls
                    raise
                errors.append(f"{getattr(client, 'model', '?')}: {e}")
                continue
            reply.calls = calls + reply.calls
            return reply
        raise LLMError("All models are busy. " + " | ".join(errors), overloaded=True, calls=calls)


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r} (only 'gemini')")
    if not settings.llm_model:
        raise ValueError("LLM_MODEL is not set")
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set")
    models = [settings.llm_model, *settings.llm_fallback_model_list()]
    clients = [
        GeminiClient(
            api_key=key,
            model=model,
            timeout_ms=settings.llm_timeout_ms,
            thinking_level=settings.llm_thinking_level,
        )
        for model in models
    ]
    return clients[0] if len(clients) == 1 else FallbackLLM(clients)
