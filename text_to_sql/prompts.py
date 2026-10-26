"""Prompt text: system prompt, business notes and few-shot examples.

Examples are written against the table names only; {schema} is filled in with
WAREHOUSE_SCHEMA so they stay correct when the star schema moves (e.g. to marts).
"""

CANNOT_ANSWER = "CANNOT_ANSWER"

# Shown next to the table in the schema context (col_description comments win if set)
TABLE_NOTES = {
    "fact_sales": "1 row = 1 invoice line (one product on one invoice)",
    "dim_customer": "1 row = 1 customer",
    "dim_product": "1 row = 1 product (stock_code)",
    "dim_date": "1 row = 1 calendar day that has sales",
}

# The dbt marts have no PK/FK constraints, so the join keys are spelled out here
COLUMN_NOTES = {
    ("fact_sales", "line_id"): "unique per row: '<invoice_no>-<line number in invoice>'",
    ("fact_sales", "customer_id"): "join to dim_customer.customer_id",
    ("fact_sales", "product_key"): "join to dim_product.product_key",
    ("fact_sales", "date_key"): "sale date, join to dim_date.date_key",
    ("fact_sales", "quantity"): "negative = returned items",
    ("fact_sales", "total_amount"): "quantity * unit_price, negative for returns",
    ("fact_sales", "is_return"): "true when quantity < 0",
    ("fact_sales", "invoice_no"): "starts with 'C' = cancellation/return invoice",
    ("dim_product", "product_key"): "md5(stock_code), join to fact_sales.product_key",
    ("dim_product", "stock_code"): "business product code",
    ("dim_date", "weekday"): "English day name, e.g. 'Monday'",
}

# Small categorical columns whose distinct values are shown so the LLM spells
# them exactly (e.g. 'EIRE', 'USA'). The only data from the DB sent to the LLM.
SAMPLE_VALUE_COLUMNS = {
    "dim_customer": ["country"],
    "dim_date": ["weekday"],
}
SAMPLE_VALUE_LIMIT = 50

BUSINESS_NOTES = """\
- Source: UCI "Online Retail" — a UK-based online gift shop, many customers are wholesalers.
- Only rows with a known customer are loaded, so customer_id is never NULL.
- Net revenue = SUM(fact_sales.total_amount) (returns are negative and reduce it).
  Gross sales = the same sum WHERE NOT is_return. Amounts are in GBP (£).
- A return / cancellation is a row with is_return = true (its invoice_no starts with 'C').
  Return rate = returned lines / all lines, unless the user asks for value.
- An "order" is a distinct invoice_no. A repeat customer has 2+ distinct non-return invoices.
- Use dim_date for year/month/weekday filters; date_key is the sale date.
- English country names as stored, e.g. 'United Kingdom' (อังกฤษ/สหราชอาณาจักร), 'EIRE' (Ireland),
  'USA', 'RSA' (South Africa)."""

SYSTEM_PROMPT = """\
You translate questions (Thai or English) about an e-commerce sales warehouse into one
PostgreSQL query.

Rules:
1. Use the PostgreSQL dialect. Always reference tables as {schema}.<table>.
2. Write exactly ONE read-only SELECT statement (WITH ... SELECT is fine). Never write
   INSERT, UPDATE, DELETE, DDL, COPY, or any other command, whatever the question says.
3. Use only the tables and columns listed in the schema below. Do not query system catalogs.
4. If the question cannot be answered from this schema (data that does not exist, or a request
   to change data), reply with {cannot_answer} followed by a one-sentence reason, and no SQL.
5. Give columns readable aliases. Add ORDER BY when the question implies a ranking or a trend.
6. Reply with the SQL inside a ```sql code block and nothing else.

Schema:
{schema_context}

Business notes:
{business_notes}

Examples:
{examples}"""

FEW_SHOT_EXAMPLES = [
    (
        "ยอดขายรายเดือนปี 2011",
        """SELECT d.month, SUM(f.total_amount) AS revenue
FROM {schema}.fact_sales f
JOIN {schema}.dim_date d ON d.date_key = f.date_key
WHERE d.year = 2011
GROUP BY d.month
ORDER BY d.month""",
    ),
    (
        "Top 10 products by quantity sold",
        """SELECT p.stock_code, p.description, SUM(f.quantity) AS units_sold
FROM {schema}.fact_sales f
JOIN {schema}.dim_product p ON p.product_key = f.product_key
WHERE NOT f.is_return
GROUP BY p.stock_code, p.description
ORDER BY units_sold DESC
LIMIT 10""",
    ),
    (
        "อัตราการคืนสินค้าแยกตามประเทศ",
        """SELECT c.country,
       COUNT(*) FILTER (WHERE f.is_return)::numeric / COUNT(*) AS return_rate
FROM {schema}.fact_sales f
JOIN {schema}.dim_customer c ON c.customer_id = f.customer_id
GROUP BY c.country
ORDER BY return_rate DESC""",
    ),
    (
        "How many customers bought more than once?",
        """SELECT COUNT(*) AS repeat_customers
FROM (
    SELECT customer_id
    FROM {schema}.fact_sales
    WHERE NOT is_return
    GROUP BY customer_id
    HAVING COUNT(DISTINCT invoice_no) >= 2
) t""",
    ),
    (
        "ลูกค้าคนไหนใช้อีเมลอะไร",
        f"{CANNOT_ANSWER} The warehouse has no customer email or contact data.",
    ),
]


def render_examples(schema: str) -> str:
    parts = []
    for question, answer in FEW_SHOT_EXAMPLES:
        answer = answer.replace("{schema}", schema)
        body = answer if answer.startswith(CANNOT_ANSWER) else f"```sql\n{answer}\n```"
        parts.append(f"Q: {question}\nA:\n{body}")
    return "\n\n".join(parts)


def build_system_prompt(schema: str, schema_context: str) -> str:
    return SYSTEM_PROMPT.format(
        schema=schema,
        cannot_answer=CANNOT_ANSWER,
        schema_context=schema_context,
        business_notes=BUSINESS_NOTES,
        examples=render_examples(schema),
    )


def build_user_prompt(
    question: str, previous_sql: str | None = None, previous_error: str | None = None
) -> str:
    prompt = f"Question: {question}"
    if previous_error:
        prompt += (
            "\n\nYour previous answer did not work.\n"
            f"Previous SQL:\n```sql\n{previous_sql or '(none)'}\n```\n"
            f"Error: {previous_error}\n"
            "Return a corrected query, or CANNOT_ANSWER if the question cannot be answered."
        )
    return prompt
