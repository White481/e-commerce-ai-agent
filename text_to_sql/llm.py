"""LLM clients. The agent only needs "system + user prompt in, text out", so any
provider fits behind LLMClient and tests can pass a fake one.
"""

from typing import Protocol

from text_to_sql.config import Settings


class LLMError(Exception):
    """The LLM call failed (network, quota, bad key). Not something a retry prompt fixes."""


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai  # imported here so tests without the key don't need it

        self._client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, system: str, user: str) -> str:
        from google.genai import errors, types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                # temperature 0: the same question should give the same SQL
                config=types.GenerateContentConfig(system_instruction=system, temperature=0),
            )
        except errors.APIError as e:
            raise LLMError(f"LLM request failed ({e.code})") from e
        return response.text or ""


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r} (only 'gemini')")
    if not settings.llm_model:
        raise ValueError("LLM_MODEL is not set")
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set")
    return GeminiClient(api_key=key, model=settings.llm_model)
