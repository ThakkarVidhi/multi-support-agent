"""SQLite client: connection, schema, read-only query."""
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a connection to the SQLite DB."""
    if db_path is None:
        from src.config import DB_PATH
        db_path = DB_PATH
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def get_schema(db_path: Path | None = None) -> str:
    """Return table schema with column names and usage hints for the agent."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='support_tickets'"
        )
        if cur.fetchone() is None:
            return "Table support_tickets not found."
        cur = conn.execute("PRAGMA table_info(support_tickets)")
        rows = cur.fetchall()
        # row: (cid, name, type, notnull, default, pk)
        cols = [row[1] for row in rows]
        # Describe columns for search: which to use for name, email, product, etc.
        return (
            "Table: support_tickets\n"
            "Columns (use these exact names): "
            + ", ".join(cols)
            + "\n\n"
            "Search hints: "
            "customer_name (text, use LIKE '%value%' for partial name); "
            "customer_email (text, use = or LIKE for email); "
            "product_purchased (text, use LIKE '%value%' for product); "
            "ticket_id, ticket_status, ticket_priority, ticket_type, ticket_subject, ticket_description; "
            "date_of_purchase, resolution, ticket_channel, customer_age, customer_gender, "
            "first_response_time, time_to_resolution, customer_satisfaction_rating."
        )
    finally:
        conn.close()


def run_query(sql: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Execute a read-only SELECT and return rows as list of dicts."""
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        logger.warning("Rejected non-SELECT query")
        return [{"error": "Only SELECT queries are allowed."}]
    conn = get_connection(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.exception("SQL error")
        return [{"error": str(e)}]
    finally:
        conn.close()
