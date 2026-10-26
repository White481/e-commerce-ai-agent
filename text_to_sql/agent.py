"""The agent loop: question -> LLM -> validator -> executor, retrying with the error.

Only errors the LLM can fix are sent back to it (validator rejections, SQL mistakes).
Timeouts, permission errors and an unreachable DB stop the loop at once.
"""

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from text_to_sql import prompts
from text_to_sql.executor import NonRetryableError, QueryExecutor, RetryableError
from text_to_sql.llm import LLMClient, LLMError
from text_to_sql.schema_provider import SchemaProvider
from text_to_sql.validator import DEFAULT_BLOCKED_FUNCTIONS, SQLValidator

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

Status = Literal["ok", "cannot_answer", "error"]


@dataclass
class Attempt:
    sql: str | None
    error: str | None = None


@dataclass
class Answer:
    status: Status
    question: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    message: str | None = None
    attempts: list[Attempt] = field(default_factory=list)


class QueryLog:
    """One JSON line per question. Logs SQL and errors, never the result rows."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, answer: Answer, duration_ms: int) -> None:
        if self.path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": answer.question,
            "status": answer.status,
            "sql": answer.sql,
            "row_count": answer.row_count,
            "message": answer.message,
            "attempts": [asdict(a) for a in answer.attempts],
            "duration_ms": duration_ms,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class TextToSQLAgent:
    def __init__(
        self,
        llm: LLMClient,
        schema_provider: SchemaProvider,
        executor: QueryExecutor,
        schema: str,
        max_attempts: int = 3,
        blocked_functions: tuple[str, ...] | list[str] = DEFAULT_BLOCKED_FUNCTIONS,
        query_log: QueryLog | None = None,
    ):
        self.llm = llm
        self.schema_provider = schema_provider
        self.executor = executor
        self.schema = schema
        self.max_attempts = max_attempts
        self.blocked_functions = blocked_functions
        self.query_log = query_log or QueryLog(None)

    def ask(self, question: str) -> Answer:
        started = time.monotonic()
        answer = self._ask(question.strip())
        self.query_log.write(answer, int((time.monotonic() - started) * 1000))
        return answer

    def _ask(self, question: str) -> Answer:
        # Built per question so a schema refresh() is picked up
        validator = SQLValidator(
            self.schema,
            self.schema_provider.table_names,
            self.executor.row_limit,
            self.blocked_functions,
        )
        system = prompts.build_system_prompt(self.schema, self.schema_provider.get_context())
        attempts: list[Attempt] = []
        previous_sql = previous_error = None

        for _ in range(self.max_attempts):
            user = prompts.build_user_prompt(question, previous_sql, previous_error)
            try:
                reply = self.llm.generate(system, user)
            except LLMError as e:
                return Answer("error", question, message=str(e), attempts=attempts)

            reason = cannot_answer_reason(reply)
            if reason is not None:
                return Answer("cannot_answer", question, message=reason, attempts=attempts)

            sql = extract_sql(reply)
            result = validator.validate(sql)
            if not result.ok:
                attempts.append(Attempt(sql, result.error))
                previous_sql, previous_error = sql, result.error
                continue

            try:
                rows = self.executor.run(result.sql)
            except RetryableError as e:
                attempts.append(Attempt(result.sql, str(e)))
                previous_sql, previous_error = result.sql, str(e)
                continue
            except NonRetryableError as e:
                attempts.append(Attempt(result.sql, str(e)))
                return Answer("error", question, sql=result.sql, message=str(e), attempts=attempts)

            attempts.append(Attempt(result.sql))
            return Answer(
                "ok",
                question,
                sql=result.sql,
                columns=rows.columns,
                rows=rows.rows,
                row_count=rows.row_count,
                truncated=rows.truncated,
                attempts=attempts,
            )

        return Answer(
            "error",
            question,
            sql=previous_sql,
            message=f"No working query after {self.max_attempts} attempts. Last error: {previous_error}",
            attempts=attempts,
        )


def cannot_answer_reason(reply: str) -> str | None:
    """The reason after CANNOT_ANSWER, or None if the reply is an answer."""
    text = reply.strip()
    if not text.startswith(prompts.CANNOT_ANSWER):
        return None
    reason = text[len(prompts.CANNOT_ANSWER):].strip(" :.-\n")
    return reason or "The question cannot be answered from this data."


def extract_sql(reply: str) -> str:
    """SQL from the first ```sql block, or the whole reply if there is no block."""
    match = _SQL_BLOCK.search(reply)
    return (match.group(1) if match else reply).strip()
