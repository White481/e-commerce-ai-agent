-- Read-only login for the Text-to-SQL agent. Run once as the admin user: make agent-role
-- Idempotent: re-running keeps the role and resets password + grants.
--
-- psql variables (the password stays outside Git, it comes from .env):
--   agent_user      e.g. analyst_ro
--   agent_password
--   schema          warehouse schema the agent may read (WAREHOUSE_SCHEMA)
--   db              database name (POSTGRES_DB)

SELECT format('CREATE ROLE %I LOGIN', :'agent_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'agent_user')
\gexec

ALTER ROLE :"agent_user" LOGIN PASSWORD :'agent_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

-- Start from nothing, then grant only what the agent needs.
-- (REVOKE first: if the warehouse IS public, the GRANT below adds USAGE back.)
REVOKE ALL ON SCHEMA public FROM :"agent_user";
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM :"agent_user";

GRANT CONNECT ON DATABASE :"db" TO :"agent_user";
GRANT USAGE ON SCHEMA :"schema" TO :"agent_user";
GRANT SELECT ON ALL TABLES IN SCHEMA :"schema" TO :"agent_user";
-- Tables created later in the schema by the current (admin) user are readable too
ALTER DEFAULT PRIVILEGES IN SCHEMA :"schema" GRANT SELECT ON TABLES TO :"agent_user";

-- Session defaults. Not a security boundary on their own (a session can SET them
-- back), the executor also sets READ ONLY + statement_timeout per transaction.
ALTER ROLE :"agent_user" SET statement_timeout = '5s';
ALTER ROLE :"agent_user" SET default_transaction_read_only = on;
ALTER ROLE :"agent_user" SET search_path = :"schema";
