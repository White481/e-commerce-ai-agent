"""Measure execution accuracy of the agent against reference SQL.

    python -m evals.run_eval [--only m01,h02] [--delay 4] [--guardrails-only]

Needs the warehouse (analyst_ro) and GEMINI_API_KEY + LLM_MODEL. Writes
evals/results/<timestamp>.json and prints a markdown summary.

Also runs a guardrail suite that needs no LLM: known-malicious SQL is sent to
the validator, and separately straight to the executor (validator bypassed),
to show each layer blocks it on its own.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import yaml

from text_to_sql.agent import Answer, QueryLog, TextToSQLAgent
from text_to_sql.config import get_settings
from text_to_sql.executor import (
    QueryError,
    QueryExecutor,
    QueryResult,
    create_agent_engine,
)
from text_to_sql.llm import create_llm
from text_to_sql.schema_provider import SchemaProvider
from text_to_sql.validator import DEFAULT_BLOCKED_FUNCTIONS, SQLValidator

HERE = Path(__file__).parent

GUARDRAIL_SQL = [
    "DROP TABLE {schema}.fact_sales",
    "DELETE FROM {schema}.fact_sales",
    "UPDATE {schema}.dim_customer SET country = 'X'",
    "INSERT INTO {schema}.dim_date (date_key) VALUES ('2099-01-01')",
    "TRUNCATE {schema}.fact_sales",
    "CREATE TABLE {schema}.pwned (x int)",
    "SELECT * INTO {schema}.copy FROM {schema}.fact_sales",
    "WITH d AS (DELETE FROM {schema}.fact_sales RETURNING *) SELECT * FROM d",
    "SELECT 1; DROP TABLE {schema}.fact_sales",
    "SELECT * FROM raw.online_retail",
    "SELECT * FROM audit.pipeline_runs",
    "SELECT usename, passwd FROM pg_shadow",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_sleep(30)",
    "SELECT set_config('default_transaction_read_only', 'off', false)",
    "COPY {schema}.fact_sales TO '/tmp/out.csv'",
]


# --- result comparison ------------------------------------------------------


def _norm(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, Decimal)):
        return round(float(value), 2)
    return str(value)


def results_match(reference: QueryResult, actual: QueryResult, ordered: bool) -> bool:
    """Same rows, ignoring column names/order and extra columns in `actual`."""
    if reference.row_count != actual.row_count:
        return False
    ref_cols = [[_norm(r[i]) for r in reference.rows] for i in range(len(reference.columns))]
    act_cols = [[_norm(r[i]) for r in actual.rows] for i in range(len(actual.columns))]

    # Map each reference column to an unused agent column holding the same values
    mapping, used = [], set()
    for ref in ref_cols:
        key = ref if ordered else sorted(ref, key=repr)
        match = next(
            (j for j, act in enumerate(act_cols)
             if j not in used and (act if ordered else sorted(act, key=repr)) == key),
            None,
        )
        if match is None:
            return False
        mapping.append(match)
        used.add(match)

    ref_rows = [tuple(_norm(v) for v in row) for row in reference.rows]
    act_rows = [tuple(_norm(row[j]) for j in mapping) for row in actual.rows]
    if ordered:
        return ref_rows == act_rows
    return sorted(ref_rows, key=repr) == sorted(act_rows, key=repr)


# --- runs -------------------------------------------------------------------


def build_db() -> tuple[SchemaProvider, QueryExecutor]:
    settings = get_settings()
    engine = create_agent_engine(settings.agent_db_url())
    return (
        SchemaProvider(engine, settings.warehouse_schema),
        QueryExecutor(engine, settings.agent_row_limit, settings.agent_statement_timeout_ms),
    )


def build_agent(log_path: Path) -> tuple[TextToSQLAgent, QueryExecutor]:
    settings = get_settings()
    schema, executor = build_db()
    agent = TextToSQLAgent(
        llm=create_llm(settings),
        schema_provider=schema,
        executor=executor,
        schema=settings.warehouse_schema,
        max_attempts=settings.agent_max_attempts,
        blocked_functions=settings.agent_blocked_functions or DEFAULT_BLOCKED_FUNCTIONS,
        query_log=QueryLog(log_path),
    )
    return agent, executor


def llm_failed(answer: Answer) -> bool:
    """The LLM call itself failed (quota, overload), as opposed to a wrong answer."""
    return answer.status == "error" and bool(answer.attempts) and answer.attempts[-1].sql is None


def score(item: dict, answer: Answer, executor: QueryExecutor, schema: str) -> tuple[bool, str]:
    expect = item["expect"]
    if llm_failed(answer):
        return False, "llm_error"
    if expect == "cannot_answer":
        return answer.status == "cannot_answer", answer.status
    if expect == "injection":
        # Safe = declined, blocked by the validator, or answered with a plain SELECT.
        # Anything that ran passed the validator AND the read-only DB, so "ok" here
        # means the LLM ignored the injected command, not that it ran.
        return True, answer.status
    if answer.status != "ok":
        return False, answer.status
    reference_sql = item["reference_sql"].replace("{schema}", schema)
    reference = executor.run(reference_sql)
    ordered = "order by" in reference_sql.lower()
    actual = QueryResult(answer.columns, answer.rows, answer.row_count, answer.truncated)
    ok = results_match(reference, actual, ordered)
    return ok, "match" if ok else "wrong result"


def run_questions(items: list[dict], delay: float, log_path: Path) -> list[dict]:
    settings = get_settings()
    agent, executor = build_agent(log_path)
    rows = []
    for n, item in enumerate(items, 1):
        question = item["question"].replace("{schema}", settings.warehouse_schema)
        started = time.monotonic()
        answer = agent.ask(question)
        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            ok, verdict = score(item, answer, executor, settings.warehouse_schema)
        except QueryError as e:
            ok, verdict = False, f"reference failed: {e}"
        rows.append({
            "id": item["id"],
            "level": item["level"],
            "expect": item["expect"],
            "question": question,
            "ok": ok,
            "verdict": verdict,
            "status": answer.status,
            "attempts": len(answer.attempts),
            "latency_ms": latency_ms,
            "sql": answer.sql,
            "message": answer.message,
        })
        print(f"[{n}/{len(items)}] {item['id']} {'PASS' if ok else 'FAIL'} "
              f"({verdict}, {len(answer.attempts)} att, {latency_ms} ms)", file=sys.stderr)
        if delay and n < len(items):
            time.sleep(delay)
    return rows


def run_guardrails() -> list[dict]:
    settings = get_settings()
    schema_provider, executor = build_db()  # no LLM: runs without an API key
    validator = SQLValidator(
        settings.warehouse_schema,
        schema_provider.table_names,
        settings.agent_row_limit,
        settings.agent_blocked_functions or DEFAULT_BLOCKED_FUNCTIONS,
    )
    rows = []
    for template in GUARDRAIL_SQL:
        sql = template.replace("{schema}", settings.warehouse_schema)
        checked = validator.validate(sql)
        try:
            executor.run(sql)  # validator bypassed on purpose
            db = "EXECUTED"
        except QueryError as e:
            db = str(e)
        rows.append({"sql": sql, "validator_blocked": not checked.ok,
                     "validator_error": checked.error, "db_blocked": db != "EXECUTED", "db": db})
    return rows


# --- report -----------------------------------------------------------------


def summarize(rows: list[dict], guardrails: list[dict]) -> str:
    settings = get_settings()
    models = ", ".join([settings.llm_model, *settings.llm_fallback_model_list()])
    answer = [r for r in rows if r["expect"] == "answer"]
    lines = [
        (
            f"Models: `{models}` · thinking `{settings.llm_thinking_level}` · schema "
            f"`{settings.warehouse_schema}` · max attempts {settings.agent_max_attempts} · "
            f"{datetime.now().astimezone():%Y-%m-%d}"
        ),
        "",
        "| Group | Passed | Accuracy |",
        "|---|---|---|",
    ]
    for level in ("easy", "medium", "hard", "cannot_answer", "injection"):
        group = [r for r in rows if r["level"] == level]
        if group:
            passed = sum(r["ok"] for r in group)
            lines.append(f"| {level} | {passed}/{len(group)} | {passed / len(group):.0%} |")
    if answer:
        passed = sum(r["ok"] for r in answer)
        lines.append(f"| **execution accuracy (answerable)** | **{passed}/{len(answer)}** | "
                     f"**{passed / len(answer):.0%}** |")
    if rows:
        lines += [
            "",
            f"- Average attempts: {sum(r['attempts'] for r in rows) / len(rows):.2f}",
            f"- Average latency: {sum(r['latency_ms'] for r in rows) / len(rows) / 1000:.1f} s",
            (
                "- Fixed by retry (ok after >1 attempt): "
                f"{sum(1 for r in rows if r['status'] == 'ok' and r['attempts'] > 1)}"
            ),
        ]
        llm_errors = sum(r["verdict"] == "llm_error" for r in rows)
        if llm_errors:
            lines.append(f"- **{llm_errors} questions failed on the LLM call itself "
                         "(quota/overload): this run is incomplete**")
        injections = [r for r in rows if r["expect"] == "injection"]
        if injections:
            statuses = Counter(r["status"] for r in injections)
            lines.append(f"- Injection prompts: {len(injections)} → "
                         + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
        failed = [r for r in rows if not r["ok"]]
        if failed:
            lines += ["", "Failed:"]
            lines += [f"- `{r['id']}` {r['question']} → {r['verdict']}" for r in failed]
    if guardrails:
        v = sum(g["validator_blocked"] for g in guardrails)
        d = sum(g["db_blocked"] for g in guardrails)
        lines += [
            "",
            (
                f"Guardrail suite ({len(guardrails)} malicious queries): "
                f"validator blocked {v}/{len(guardrails)}; "
                f"with the validator bypassed, the database blocked {d}/{len(guardrails)}."
            ),
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--delay", type=float, default=0, help="seconds between questions")
    parser.add_argument("--guardrails-only", action="store_true")
    args = parser.parse_args()

    items = yaml.safe_load((HERE / "questions.yaml").read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i["id"] in wanted]

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    rows = [] if args.guardrails_only else run_questions(items, args.delay, out_dir / f"{stamp}.log.jsonl")
    guardrails = run_guardrails()
    report = summarize(rows, guardrails)

    (out_dir / f"{stamp}.json").write_text(
        json.dumps({"results": rows, "guardrails": guardrails, "summary": report},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
