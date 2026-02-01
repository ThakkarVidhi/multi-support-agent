"""Tests for SQLite client schema (uses project DB if present)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.db import sqlite_client

DB_PATH = ROOT / "data" / "customer_support.db"


@pytest.mark.skipif(not DB_PATH.exists(), reason="DB not seeded")
def test_get_schema_returns_table_and_columns():
    """Schema string includes support_tickets and column names."""
    schema = sqlite_client.get_schema(DB_PATH)
    assert "support_tickets" in schema
    assert "customer_name" in schema
    assert "ticket_id" in schema
    assert "customer_email" in schema
    assert "product_purchased" in schema
    assert "ticket_status" in schema


@pytest.mark.skipif(not DB_PATH.exists(), reason="DB not seeded")
def test_run_query_select_only():
    """Only SELECT is allowed."""
    rows = sqlite_client.run_query("INSERT INTO support_tickets (ticket_id) VALUES (1)", DB_PATH)
    assert len(rows) == 1 and "error" in rows[0]


@pytest.mark.skipif(not DB_PATH.exists(), reason="DB not seeded")
def test_run_query_valid_select():
    """Valid SELECT returns list of dicts."""
    rows = sqlite_client.run_query("SELECT ticket_id, customer_name FROM support_tickets LIMIT 1", DB_PATH)
    assert isinstance(rows, list)
    if rows and "error" not in (rows[0] or {}):
        assert isinstance(rows[0], dict)
        assert "ticket_id" in rows[0] or "customer_name" in rows[0]
