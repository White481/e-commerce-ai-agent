"""Run validated SQL as the read-only agent role.

Each query runs in its own READ ONLY transaction with a statement_timeout, on
top of the role's own privileges. DB errors are split into ones the LLM can
fix (retryable) and ones it can't (timeout, permission, connection).
"""

import datetime as dt
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

# SQLSTATE classes that mean "the SQL itself is wrong": the LLM can fix these.
# 42 = syntax error / undefined object, 22 = data exception (bad cast, division by zero)
_RETRYABLE_CLASSES = ("42", "22", "21")
_PERMISSION_DENIED = "42501"
_QUERY_CANCELED = "57014"  # statement_timeout
_READ_ONLY_TRANSACTION = "25006"


class QueryError(Exception):
    pass


class RetryableError(QueryError):
    """The query is wrong (syntax, unknown column...): send the message back to the LLM."""


class NonRetryableError(QueryError):
    """Timeout, permission denied, DB down: retrying with another query won't help."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool


def create_agent_engine(url, pool_size: int = 5) -> Engine:
    engine = create_engine(url, pool_pre_ping=True, pool_size=pool_size)

    @event.listens_for(engine, "checkin")
    def _reset_session(dbapi_conn, _record):
        # Any role may change its own session settings (e.g. set_config(...,
        # false)); don't let that leak to the next query on this pooled connection.
        # RESET ALL is transactional, hence the commit.
        if dbapi_conn is None:
            return
        with dbapi_conn.cursor() as cursor:
            cursor.execute("RESET ALL")
        dbapi_conn.commit()

    return engine


class QueryExecutor:
    def __init__(self, engine: Engine, row_limit: int, timeout_ms: int):
        self.engine = engine
        self.row_limit = row_limit
        self.timeout_ms = int(timeout_ms)

    def run(self, sql: str) -> QueryResult:
        try:
            with self.engine.connect() as conn, conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text(f"SET LOCAL statement_timeout = {self.timeout_ms}"))
                # Raw DBAPI cursor without parameters: psycopg then leaves '%' and
                # ':name' in the SQL alone (SQLAlchemy text() would treat them as binds)
                cursor = conn.connection.cursor()
                try:
                    cursor.execute(sql)
                    if cursor.description is None:
                        raise NonRetryableError("Query returned no result set.")
                    columns = [col.name for col in cursor.description]
                    fetched = cursor.fetchmany(self.row_limit + 1)  # +1: detect truncation
                finally:
                    cursor.close()
        except QueryError:
            raise
        except Exception as e:
            raise _classify(e) from e

        rows = [[_to_json(v) for v in row] for row in fetched[: self.row_limit]]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=len(fetched) > self.row_limit,
        )

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))


def _classify(error: Exception) -> QueryError:
    """Map a driver error to a message that is safe to show (no connection string)."""
    original = error.orig if isinstance(error, DBAPIError) else error
    sqlstate = getattr(original, "sqlstate", None) or ""
    diag = getattr(original, "diag", None)
    message = (getattr(diag, "message_primary", None) or str(original).splitlines()[0]).strip()
    hint = getattr(diag, "message_hint", None)
    if hint:
        message = f"{message} (hint: {hint})"

    if sqlstate == _QUERY_CANCELED:
        return NonRetryableError("Query timed out. Try a narrower question (e.g. a shorter date range).")
    if sqlstate in (_PERMISSION_DENIED, _READ_ONLY_TRANSACTION):
        return NonRetryableError(f"Not allowed: {message}")
    if sqlstate[:2] in _RETRYABLE_CLASSES:
        return RetryableError(message)
    if isinstance(error, SQLAlchemyError) and not sqlstate:
        return NonRetryableError("Database is unavailable.")
    return NonRetryableError("Query failed.")


def _to_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return "<binary>"
    return value
