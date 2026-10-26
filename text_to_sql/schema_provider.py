"""Read the warehouse schema from the catalog and format it for the LLM.

Runs as the agent role, so information_schema only shows tables it can SELECT.
The result is cached in memory until refresh() is called.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

from text_to_sql import prompts

_COLUMNS_SQL = text("""
    SELECT c.table_name, c.column_name, c.data_type, c.is_nullable = 'YES' AS nullable,
           col_description(format('%I.%I', c.table_schema, c.table_name)::regclass,
                           c.ordinal_position) AS comment,
           obj_description(format('%I.%I', c.table_schema, c.table_name)::regclass,
                           'pg_class') AS table_comment
    FROM information_schema.columns c
    JOIN information_schema.tables t
      ON t.table_schema = c.table_schema AND t.table_name = c.table_name
    WHERE c.table_schema = :schema AND t.table_type = 'BASE TABLE'
    ORDER BY c.table_name, c.ordinal_position
""")

# PK and FK columns of tables in the schema (single-column keys are all this warehouse uses)
_KEYS_SQL = text("""
    SELECT rel.relname AS table_name, att.attname AS column_name, con.contype,
           frel.relname AS ref_table, fatt.attname AS ref_column
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    JOIN pg_namespace ns ON ns.oid = rel.relnamespace
    JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = con.conkey[1]
    LEFT JOIN pg_class frel ON frel.oid = con.confrelid
    LEFT JOIN pg_attribute fatt ON fatt.attrelid = con.confrelid AND fatt.attnum = con.confkey[1]
    WHERE ns.nspname = :schema AND con.contype IN ('p', 'f')
""")


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool
    comment: str | None = None
    primary_key: bool = False
    references: str | None = None


@dataclass
class Table:
    name: str
    comment: str | None = None
    columns: list[Column] = field(default_factory=list)


class SchemaProvider:
    def __init__(self, engine: Engine, schema: str):
        self.engine = engine
        self.schema = schema
        self._lock = threading.Lock()
        self._tables: dict[str, Table] | None = None
        self._context: str | None = None

    @property
    def table_names(self) -> set[str]:
        self._ensure_loaded()
        return set(self._tables)

    def get_context(self) -> str:
        self._ensure_loaded()
        return self._context

    def refresh(self) -> str:
        with self._lock:
            self._tables, self._context = self._load()
        return self._context

    def _ensure_loaded(self) -> None:
        if self._tables is None:
            with self._lock:
                if self._tables is None:
                    self._tables, self._context = self._load()

    def _load(self) -> tuple[dict[str, Table], str]:
        with self.engine.connect() as conn:
            column_rows = conn.execute(_COLUMNS_SQL, {"schema": self.schema}).all()
            key_rows = conn.execute(_KEYS_SQL, {"schema": self.schema}).all()
            tables = _build_tables(column_rows, key_rows)
            samples = self._sample_values(conn, tables)
            date_range = self._date_range(conn, tables)
        return tables, self._format(tables, samples, date_range)

    def _sample_values(self, conn, tables: dict[str, Table]) -> dict[tuple[str, str], list[str]]:
        samples = {}
        for table, columns in prompts.SAMPLE_VALUE_COLUMNS.items():
            if table not in tables:
                continue
            known = {c.name for c in tables[table].columns}
            for column in columns:
                if column not in known:
                    continue
                # Identifiers come from the catalog + our own config, quoted with %I-style quoting
                sql = text(
                    f"SELECT DISTINCT {_ident(column)} FROM {_ident(self.schema)}.{_ident(table)} "
                    f"WHERE {_ident(column)} IS NOT NULL ORDER BY 1 LIMIT :n"
                )
                values = conn.execute(sql, {"n": prompts.SAMPLE_VALUE_LIMIT + 1}).scalars().all()
                samples[(table, column)] = [str(v) for v in values]
        return samples

    def _date_range(self, conn, tables: dict[str, Table]) -> tuple[str, str] | None:
        if "dim_date" not in tables or "date_key" not in {c.name for c in tables["dim_date"].columns}:
            return None
        sql = text(f"SELECT min(date_key), max(date_key) FROM {_ident(self.schema)}.dim_date")
        low, high = conn.execute(sql).one()
        return (str(low), str(high)) if low else None

    def _format(self, tables, samples, date_range) -> str:
        lines = []
        for table in tables.values():
            note = table.comment or prompts.TABLE_NOTES.get(table.name)
            lines.append(f"TABLE {self.schema}.{table.name}" + (f"  -- {note}" if note else ""))
            for col in table.columns:
                parts = [f"  {col.name} {col.data_type.upper()}"]
                if col.primary_key:
                    parts.append("PK")
                if col.references:
                    parts.append(f"FK -> {col.references}")
                if not col.nullable and not col.primary_key:
                    parts.append("NOT NULL")
                note = col.comment or prompts.COLUMN_NOTES.get((table.name, col.name))
                values = samples.get((table.name, col.name))
                if values:
                    shown = values[: prompts.SAMPLE_VALUE_LIMIT]
                    more = ", ..." if len(values) > prompts.SAMPLE_VALUE_LIMIT else ""
                    note = (f"{note}; " if note else "") + "values: " + ", ".join(
                        f"'{v}'" for v in shown
                    ) + more
                line = " ".join(parts)
                lines.append(f"{line}  -- {note}" if note else line)
            lines.append("")
        if date_range:
            lines.append(f"Data covers {date_range[0]} to {date_range[1]} (dim_date.date_key).")
        return "\n".join(lines).strip()


def _build_tables(column_rows, key_rows) -> dict[str, Table]:
    keys = defaultdict(dict)
    for row in key_rows:
        keys[(row.table_name, row.column_name)][row.contype] = (
            f"{row.ref_table}.{row.ref_column}" if row.contype == "f" else True
        )
    tables: dict[str, Table] = {}
    for row in column_rows:
        table = tables.setdefault(row.table_name, Table(row.table_name, row.table_comment))
        key = keys.get((row.table_name, row.column_name), {})
        table.columns.append(
            Column(
                name=row.column_name,
                data_type=row.data_type,
                nullable=row.nullable,
                comment=row.comment,
                primary_key="p" in key,
                references=key.get("f"),
            )
        )
    return tables


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
