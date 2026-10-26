"""Agent settings, read from environment variables / .env.

The agent connects with its own read-only login (AGENT_DB_USER), never with the
admin POSTGRES_USER that owns the warehouse. Host/port/database are shared.
"""

from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ecommerce"
    # Read only to refuse running the agent as the admin user
    postgres_user: str | None = None

    agent_db_user: str = "analyst_ro"
    agent_db_password: SecretStr = SecretStr("")
    warehouse_schema: str = "marts"

    llm_provider: str = "gemini"
    llm_model: str = ""
    gemini_api_key: SecretStr = SecretStr("")
    # Comma-separated, tried in order when LLM_MODEL is overloaded (503/429/timeout)
    llm_fallback_models: str = ""
    llm_timeout_ms: int = 30000
    # minimal / low / medium / high; empty = the model's default
    llm_thinking_level: str = "low"

    agent_max_attempts: int = 3
    agent_row_limit: int = 200
    agent_statement_timeout_ms: int = 5000
    agent_query_log_path: str = "logs/agent_queries.jsonl"
    # Empty = use validator.DEFAULT_BLOCKED_FUNCTIONS
    agent_blocked_functions: list[str] = []

    @model_validator(mode="after")
    def _not_admin(self) -> "Settings":
        if self.postgres_user and self.agent_db_user == self.postgres_user:
            raise ValueError(
                "AGENT_DB_USER must be a dedicated read-only role, not POSTGRES_USER"
            )
        return self

    def llm_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    def agent_db_url(self) -> URL:
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.agent_db_user,
            password=self.agent_db_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
