"""HTTP API for the agent:  make agent  ->  http://localhost:8000/docs

The agent is built at startup (DB engine, schema, LLM), not at import, so
create_app(build=...) can hand tests a fake agent.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from text_to_sql.agent import QueryLog, Status, TextToSQLAgent
from text_to_sql.config import get_settings
from text_to_sql.executor import QueryExecutor, create_agent_engine
from text_to_sql.llm import create_llm
from text_to_sql.schema_provider import SchemaProvider
from text_to_sql.validator import DEFAULT_BLOCKED_FUNCTIONS


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, examples=["ยอดขายรายเดือนปี 2011"])


class AttemptOut(BaseModel):
    sql: str | None
    error: str | None
    llm_ms: int | None
    db_ms: int | None


class AskResponse(BaseModel):
    status: Status
    question: str
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    message: str | None
    attempts: list[AttemptOut]


def build_agent() -> TextToSQLAgent:
    settings = get_settings()
    engine = create_agent_engine(settings.agent_db_url())
    return TextToSQLAgent(
        llm=create_llm(settings),
        schema_provider=SchemaProvider(engine, settings.warehouse_schema),
        executor=QueryExecutor(engine, settings.agent_row_limit, settings.agent_statement_timeout_ms),
        schema=settings.warehouse_schema,
        max_attempts=settings.agent_max_attempts,
        blocked_functions=settings.agent_blocked_functions or DEFAULT_BLOCKED_FUNCTIONS,
        query_log=QueryLog(settings.agent_query_log_path),
    )


def get_agent(request: Request) -> TextToSQLAgent:
    return request.app.state.agent


AgentDep = Annotated[TextToSQLAgent, Depends(get_agent)]


def create_app(build: Callable[[], TextToSQLAgent] = build_agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = build()
        yield

    app = FastAPI(title="E-commerce Text-to-SQL", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    # Plain def handlers: DB and LLM calls block, so FastAPI runs them in its threadpool
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, agent: AgentDep):
        if not body.question.strip():
            raise HTTPException(422, "question is empty")
        return agent.ask(body.question)

    @app.get("/schema")
    def schema(agent: AgentDep) -> dict[str, str]:
        return {"schema": agent.schema, "context": agent.schema_provider.get_context()}

    @app.post("/schema/refresh")
    def refresh_schema(agent: AgentDep) -> dict[str, str]:
        return {"schema": agent.schema, "context": agent.schema_provider.refresh()}

    @app.get("/health")
    def health(agent: AgentDep) -> dict[str, str]:
        try:
            agent.executor.ping()
        except Exception as e:  # any failure means "not healthy"
            raise HTTPException(503, "database unavailable") from e
        return {"status": "ok"}

    return app


app = create_app()
