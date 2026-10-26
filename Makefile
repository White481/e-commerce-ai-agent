.PHONY: agent-role agent test lint

# Values come from .env (AGENT_DB_PASSWORD is not in Git).
# Runs psql inside the pipeline's Postgres container as its admin user.
agent-role:
	set -a && . ./.env && set +a && \
	docker exec -i "$${PG_CONTAINER:-ecommerce_postgres}" psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -v ON_ERROR_STOP=1 \
		-v agent_user="$${AGENT_DB_USER:-analyst_ro}" -v agent_password="$$AGENT_DB_PASSWORD" \
		-v schema="$${WAREHOUSE_SCHEMA:-marts}" -v db="$$POSTGRES_DB" \
		< sql/create_readonly_role.sql

agent:  # http://localhost:8000/docs
	uvicorn text_to_sql.api:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .
