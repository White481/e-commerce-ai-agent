"""Integration: the DB itself must refuse writes, even with the validator bypassed.

Needs the pipeline's Postgres (make up in e-commerce-data-pipeline) and make agent-role here.
Skipped when the database is not reachable, e.g. in CI.
"""

import pytest
from sqlalchemy.exc import OperationalError

from text_to_sql.config import Settings
from text_to_sql.executor import (
    NonRetryableError,
    QueryExecutor,
    RetryableError,
    create_agent_engine,
)
from text_to_sql.schema_provider import SchemaProvider

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings():
    return Settings()


@pytest.fixture(scope="module")
def engine(settings):
    if not settings.agent_db_password.get_secret_value():
        pytest.skip("AGENT_DB_PASSWORD not set")
    engine = create_agent_engine(settings.agent_db_url())
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip("warehouse database not reachable")
    yield engine
    engine.dispose()


@pytest.fixture
def executor(engine):
    return QueryExecutor(engine, row_limit=5, timeout_ms=1000)


def test_connects_as_agent_role_not_admin(executor, settings):
    result = executor.run("SELECT current_user")
    assert result.rows == [[settings.agent_db_user]]


def test_select_works_and_reports_truncation(executor, settings):
    result = executor.run(f"SELECT invoice_no FROM {settings.warehouse_schema}.fact_sales")
    assert result.row_count == 5
    assert result.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM {schema}.fact_sales",
        "UPDATE {schema}.dim_customer SET country = 'x'",
        "INSERT INTO {schema}.dim_date (date_key) VALUES ('2099-01-01')",
        "CREATE TABLE {schema}.agent_was_here (x int)",
        "DROP TABLE {schema}.fact_sales",
        "TRUNCATE {schema}.fact_sales",
        # Turning the session default off doesn't help: the transaction is READ ONLY
        "SET default_transaction_read_only = off; DELETE FROM {schema}.fact_sales",
    ],
)
def test_writes_fail_at_the_database(executor, settings, sql):
    with pytest.raises(NonRetryableError, match="Not allowed"):
        executor.run(sql.format(schema=settings.warehouse_schema))


def test_privileges_alone_block_writes(engine, settings):
    """Even outside the READ ONLY transaction, the role has no write privilege."""
    with engine.connect() as conn:
        conn.exec_driver_sql("SET default_transaction_read_only = off")
        # The SET ran in an autobegun transaction that was already read-only; start a new one
        conn.commit()
        with pytest.raises(Exception, match="permission denied"):
            conn.exec_driver_sql(f"DELETE FROM {settings.warehouse_schema}.fact_sales")
        conn.rollback()


# Real tables: public grants USAGE to everyone by default, so only the table privilege stops it
@pytest.mark.parametrize(
    "table",
    ["raw.online_retail", "audit.pipeline_runs", "staging.stg_online_retail", "public.fact_sales"],
)
def test_other_schemas_are_not_readable(executor, settings, table):
    if table.split(".")[0] == settings.warehouse_schema:
        pytest.skip("this is the warehouse schema")
    try:
        executor.run(f"SELECT * FROM {table}")
    except NonRetryableError as e:
        assert "permission denied" in str(e)
    except RetryableError:
        pytest.skip(f"{table} does not exist in this database")
    else:
        pytest.fail("query on another schema succeeded")


def test_slow_query_hits_statement_timeout(executor):
    with pytest.raises(NonRetryableError, match="timed out"):
        executor.run("SELECT pg_sleep(3)")


def test_sql_mistakes_are_retryable(executor, settings):
    with pytest.raises(RetryableError, match="does not exist"):
        executor.run(f"SELECT no_such_column FROM {settings.warehouse_schema}.fact_sales")


def test_percent_and_colon_in_sql_are_left_alone(executor, settings):
    result = executor.run(
        f"SELECT count(*) FROM {settings.warehouse_schema}.fact_sales "
        "WHERE invoice_no LIKE 'C%' AND ':not_a_param' <> ''"
    )
    assert result.rows[0][0] > 0


def test_schema_provider_sees_only_warehouse_tables(engine, settings):
    provider = SchemaProvider(engine, settings.warehouse_schema)
    assert {"fact_sales", "dim_customer", "dim_product", "dim_date"} <= provider.table_names
    context = provider.get_context()
    assert f"TABLE {settings.warehouse_schema}.fact_sales" in context
    assert "join to dim_customer.customer_id" in context
