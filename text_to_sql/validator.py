"""Check LLM-generated SQL on the AST (sqlglot) before it reaches the database.

This is one layer of several: the DB role and the READ ONLY transaction still
apply if something gets past here. Error messages are written for the LLM,
which receives them on retry.
"""

from dataclasses import dataclass
from fnmatch import fnmatch

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# fnmatch patterns, matched against the lower-case function name
DEFAULT_BLOCKED_FUNCTIONS = (
    "pg_sleep*",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_*",
    "pg_stat_file",
    "lo_*",
    "dblink*",
    "set_config",
    "current_setting",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_advisory*",
    "pg_notify",
    "version",
    "current_version",
    "nextval",
    "setval",
    # Run an arbitrary query given as a string: hides tables from the allowlist check
    "query_to_xml*",
    "cursor_to_xml*",
    "table_to_xml*",
    "schema_to_xml*",
    "database_to_xml*",
)

# Nodes that modify data or schema, rejected anywhere in the tree (e.g. inside a CTE)
_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Grant,
    exp.Revoke,
    exp.Command,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Lock,
)

_QUERY_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    sql: str | None = None
    error: str | None = None


class SQLValidator:
    def __init__(
        self,
        schema: str,
        allowed_tables: set[str],
        row_limit: int,
        blocked_functions: tuple[str, ...] | list[str] = DEFAULT_BLOCKED_FUNCTIONS,
    ):
        self.schema = schema.lower()
        self.allowed_tables = {t.lower() for t in allowed_tables}
        self.row_limit = row_limit
        self.blocked_functions = tuple(p.lower() for p in blocked_functions)

    def validate(self, sql: str) -> ValidationResult:
        try:
            tree = self._check(sql)
        except _Rejected as e:
            return ValidationResult(ok=False, error=str(e))
        return ValidationResult(ok=True, sql=tree.sql(dialect="postgres", comments=False))

    def _check(self, sql: str) -> exp.Expression:
        if not sql or not sql.strip():
            raise _Rejected("Empty query. Return exactly one SELECT statement.")
        try:
            statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
        except ParseError as e:
            raise _Rejected(f"SQL could not be parsed (PostgreSQL dialect): {_first_line(e)}")

        if len(statements) != 1:
            raise _Rejected(
                f"Expected exactly one SQL statement, got {len(statements)}. "
                "Return a single SELECT without ';'-separated statements."
            )
        tree = statements[0]

        if not isinstance(tree, _QUERY_ROOTS):
            raise _Rejected(
                f"Only SELECT queries are allowed, got {_kind(tree)}. "
                "Do not write INSERT/UPDATE/DELETE/DDL or other commands."
            )
        if isinstance(tree, exp.Subquery) and not isinstance(tree.unnest(), _QUERY_ROOTS):
            raise _Rejected("Only SELECT queries are allowed.")

        for node in tree.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise _Rejected(_forbidden_message(node))

        self._check_tables(tree)
        self._check_functions(tree)
        self._enforce_limit(tree)
        return tree

    def _check_tables(self, tree: exp.Expression) -> None:
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        for table in tree.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                continue  # table function such as generate_series(): checked as a function
            name = table.name.lower()
            schema = table.db.lower()
            if table.catalog:
                raise _Rejected(f"Table '{table.sql()}' uses a database prefix; use {self.schema}.<table>.")
            if not schema and name in cte_names:
                continue
            if schema in ("", self.schema) and name in self.allowed_tables:
                # Qualify so the query does not depend on the role's search_path
                table.set("db", exp.to_identifier(self.schema))
                continue
            shown = f"{schema}.{name}" if schema else name
            raise _Rejected(
                f"Table '{shown}' is not allowed. Only these tables can be queried: "
                + ", ".join(f"{self.schema}.{t}" for t in sorted(self.allowed_tables))
            )

    def _check_functions(self, tree: exp.Expression) -> None:
        for func in tree.find_all(exp.Func):
            for name in _function_names(func):
                if any(fnmatch(name, pattern) for pattern in self.blocked_functions):
                    raise _Rejected(f"Function '{name}' is not allowed.")

    def _enforce_limit(self, tree: exp.Expression) -> None:
        current = tree.args.get("limit")
        requested = _literal_int(current)
        new = self.row_limit if requested is None else min(requested, self.row_limit)
        tree.set("limit", exp.Limit(expression=exp.Literal.number(new)))


class _Rejected(Exception):
    pass


def _function_names(func: exp.Func) -> set[str]:
    """Names a function can be known by. sqlglot maps some to typed nodes with an
    empty .name (version() -> CurrentVersion), so also use the rendered name."""
    names = {func.sql_name().lower()}
    if isinstance(func, exp.Anonymous):
        names.add(func.name.lower())
    rendered = func.sql(dialect="postgres").split("(", 1)[0].strip().lower()
    if rendered:
        names.add(rendered.rsplit(".", 1)[-1])
    return names


def _literal_int(limit: exp.Expression | None) -> int | None:
    """Row count from LIMIT n / FETCH FIRST n ROWS, or None if absent or not a literal."""
    if limit is None:
        return None
    value = limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            return int(value.this)
        except ValueError:
            return None
    return None


def _kind(node: exp.Expression) -> str:
    if isinstance(node, exp.Command):
        return str(node.this).upper()
    return node.key.upper()


def _forbidden_message(node: exp.Expression) -> str:
    if isinstance(node, exp.Into):
        return "SELECT ... INTO is not allowed (it creates a table). Remove INTO."
    if isinstance(node, exp.Lock):
        return "Row locking (FOR UPDATE / FOR SHARE) is not allowed."
    return (
        f"{_kind(node)} is not allowed anywhere in the query, including inside a CTE. "
        "Only read data with SELECT."
    )


def _first_line(error: Exception) -> str:
    return str(error).splitlines()[0]
