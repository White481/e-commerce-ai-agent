"""LLM clients. The agent only needs "system + user prompt in, text out", so any
provider fits behind LLMClient and tests can pass a fake one.
"""

from typing import Protocol

from text_to_sql.config import Settings


class LLMError(Exception):
    """The LLM call failed (network, quota, bad key). Not something a retry prompt fixes."""

    def __init__(self, message: str, overloaded: bool = False):
        super().__init__(message)
        # Busy or rate-limited (429/5xx/timeout): another model may still answer
        self.overloaded = overloaded


_OVERLOADED_CODES = {429, 500, 503, 504}


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_ms: int = 30_000, thinking_level: str = "low"):
        from google import genai  # imported here so tests without the key don't need it
        from google.genai import types

        # 429 / 5xx (e.g. 503 "model overloaded") are common and short-lived: back off and retry.
        # The timeout is per request, so one stuck call can't hold /ask for minutes.
        retry = types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=4.0)
        options = types.HttpOptions(timeout=timeout_ms, retry_options=retry)
        self._client = genai.Client(api_key=api_key, http_options=options)
        self.model = model
        self.timeout_ms = timeout_ms
        # Gemini 3 thinks by default, which took minutes per question; SQL over 4 tables needs little
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level.upper()) if thinking_level else None

    def generate(self, system: str, user: str) -> str:
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
            raise LLMError(
                f"LLM request failed ({e.code} {e.status}): {e.message}",
                overloaded=e.code in _OVERLOADED_CODES,
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

    def generate(self, system: str, user: str) -> str:
        errors = []
        for client in self.clients:
            try:
                return client.generate(system, user)
            except LLMError as e:
                if not e.overloaded:
                    raise
                errors.append(f"{getattr(client, 'model', '?')}: {e}")
        raise LLMError("All models are busy. " + " | ".join(errors), overloaded=True)


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
