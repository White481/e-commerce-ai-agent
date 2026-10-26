"""Agent loop and API with a fake LLM, schema and executor: no DB or API key needed."""

import json

import pytest
from fastapi.testclient import TestClient

from text_to_sql.agent import (
    QueryLog,
    TextToSQLAgent,
    cannot_answer_reason,
    extract_sql,
)
from text_to_sql.api import create_app
from text_to_sql.executor import NonRetryableError, QueryResult, RetryableError
from text_to_sql.llm import LLMError

TABLES = {"fact_sales", "dim_customer", "dim_product", "dim_date"}


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeSchema:
    table_names = TABLES

    def get_context(self):
        return "TABLE marts.fact_sales\n  total_amount NUMERIC"

    def refresh(self):
        return self.get_context()


class FakeExecutor:
    row_limit = 200

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.ran = []
        self.healthy = True

    def run(self, sql):
        self.ran.append(sql)
        outcome = self.outcomes.pop(0) if self.outcomes else QueryResult(["n"], [[1]], 1, False)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def ping(self):
        if not self.healthy:
            raise ConnectionError("down")


def make_agent(llm, executor=None, **kwargs):
    return TextToSQLAgent(
        llm=llm,
        schema_provider=FakeSchema(),
        executor=executor or FakeExecutor(),
        schema="marts",
        **kwargs,
    )


def sql_reply(sql):
    return f"```sql\n{sql}\n```"


def test_answers_on_the_first_try():
    executor = FakeExecutor(QueryResult(["revenue"], [[9769872.05]], 1, False))
    llm = FakeLLM(sql_reply("SELECT SUM(total_amount) AS revenue FROM fact_sales"))

    answer = make_agent(llm, executor).ask("  ยอดขายรวม  ")

    assert answer.status == "ok"
    assert answer.question == "ยอดขายรวม"
    assert answer.rows == [[9769872.05]]
    # The validator qualified the table and added the LIMIT before it ran
    assert executor.ran == ["SELECT SUM(total_amount) AS revenue FROM marts.fact_sales LIMIT 200"]
    assert len(answer.attempts) == 1 and answer.attempts[0].error is None
    assert "marts.fact_sales" in llm.calls[0][0]  # schema context is in the system prompt


def test_validator_rejection_is_sent_back_and_fixed():
    llm = FakeLLM(
        sql_reply("DELETE FROM fact_sales"),
        sql_reply("SELECT COUNT(*) FROM fact_sales"),
    )
    executor = FakeExecutor()

    answer = make_agent(llm, executor).ask("delete everything")

    assert answer.status == "ok"
    assert len(executor.ran) == 1  # the DELETE never reached the database
    assert "Only SELECT" in answer.attempts[0].error
    retry_prompt = llm.calls[1][1]
    assert "DELETE FROM fact_sales" in retry_prompt and "Only SELECT" in retry_prompt


def test_retryable_db_error_is_sent_back_and_fixed():
    llm = FakeLLM(sql_reply("SELECT revenue FROM fact_sales"), sql_reply("SELECT total_amount FROM fact_sales"))
    executor = FakeExecutor(RetryableError('column "revenue" does not exist'))

    answer = make_agent(llm, executor).ask("revenue")

    assert answer.status == "ok"
    assert len(answer.attempts) == 2
    assert 'column "revenue" does not exist' in llm.calls[1][1]


def test_non_retryable_error_stops_at_once():
    llm = FakeLLM(sql_reply("SELECT * FROM fact_sales"), sql_reply("never used"))
    executor = FakeExecutor(NonRetryableError("Query timed out."))

    answer = make_agent(llm, executor).ask("everything")

    assert answer.status == "error"
    assert answer.message == "Query timed out."
    assert len(llm.calls) == 1


def test_gives_up_after_max_attempts():
    llm = FakeLLM(*[sql_reply("SELECT * FROM pg_user")] * 3)

    answer = make_agent(llm, max_attempts=3).ask("users")

    assert answer.status == "error"
    assert len(answer.attempts) == 3
    assert answer.message.startswith("No working query after 3 attempts")
    assert "not allowed" in answer.message


def test_cannot_answer_is_passed_through():
    executor = FakeExecutor()
    answer = make_agent(FakeLLM("CANNOT_ANSWER The warehouse has no email data."), executor).ask("emails?")

    assert answer.status == "cannot_answer"
    assert answer.message == "The warehouse has no email data"
    assert executor.ran == []


def test_llm_failure_is_an_error_answer():
    answer = make_agent(FakeLLM(LLMError("LLM request failed (429)"))).ask("x")
    assert answer.status == "error"
    assert answer.message == "LLM request failed (429)"


def test_query_log_has_sql_but_not_rows(tmp_path):
    log = tmp_path / "logs" / "q.jsonl"
    executor = FakeExecutor(QueryResult(["country"], [["SECRET-ROW"]], 1, False))
    agent = make_agent(FakeLLM(sql_reply("SELECT country FROM dim_customer")), executor, query_log=QueryLog(log))

    agent.ask("ประเทศ")

    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["question"] == "ประเทศ"
    assert record["status"] == "ok"
    assert record["sql"].startswith("SELECT country FROM marts.dim_customer")
    assert "SECRET-ROW" not in log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("reply", "sql"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("Here you go:\n```SQL\nSELECT 2\n```\nDone.", "SELECT 2"),
        ("```\nSELECT 3\n```", "SELECT 3"),
        ("SELECT 4", "SELECT 4"),
    ],
)
def test_extract_sql(reply, sql):
    assert extract_sql(reply) == sql


def test_cannot_answer_reason():
    assert cannot_answer_reason("SELECT 1") is None
    assert cannot_answer_reason("CANNOT_ANSWER") == "The question cannot be answered from this data."
    assert cannot_answer_reason("CANNOT_ANSWER: no such data.") == "no such data"


@pytest.fixture
def client_and_executor():
    executor = FakeExecutor()
    llm = FakeLLM(*[sql_reply("SELECT COUNT(*) AS n FROM fact_sales")] * 5)
    app = create_app(build=lambda: make_agent(llm, executor))
    with TestClient(app) as client:
        yield client, executor


def test_api_ask(client_and_executor):
    client, _ = client_and_executor
    response = client.post("/ask", json={"question": "มีกี่แถว"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["columns"] == ["n"] and body["rows"] == [[1]]
    assert body["sql"] == "SELECT COUNT(*) AS n FROM marts.fact_sales LIMIT 200"


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_api_rejects_bad_questions(client_and_executor, question):
    client, executor = client_and_executor
    assert client.post("/ask", json={"question": question}).status_code == 422
    assert executor.ran == []


def test_api_health(client_and_executor):
    client, executor = client_and_executor
    assert client.get("/health").json() == {"status": "ok"}
    executor.healthy = False
    assert client.get("/health").status_code == 503


def test_api_root_redirects_to_docs(client_and_executor):
    client, _ = client_and_executor
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_api_schema(client_and_executor):
    client, _ = client_and_executor
    assert client.get("/schema").json()["schema"] == "marts"


def test_attempts_record_llm_and_db_time():
    answer = make_agent(FakeLLM(sql_reply("SELECT 1"))).ask("x")
    assert answer.attempts[0].llm_ms is not None and answer.attempts[0].db_ms is not None


def test_gemini_client_sends_thinking_level_and_timeout():
    from text_to_sql.llm import GeminiClient

    client = GeminiClient(api_key="test-key", model="m", timeout_ms=1234, thinking_level="low")
    sent = {}

    def fake_generate_content(**kwargs):
        sent.update(kwargs)
        return type("Response", (), {"text": "```sql\nSELECT 1\n```"})()

    client._client.models.generate_content = fake_generate_content
    assert client.generate("system", "user") == "```sql\nSELECT 1\n```"
    assert sent["config"].thinking_config.thinking_level.value == "LOW"
    assert sent["config"].temperature == 0
    assert client._client._api_client._http_options.timeout == 1234
