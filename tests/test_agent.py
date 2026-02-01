"""Tests for agent response structure and routing (no live LLM/DB)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.agent.agent import AgentResponse, _parse_sql_tool_output


def test_parse_sql_tool_output_with_result():
    """Parse SQL and result from tool output."""
    out = "SQL: SELECT * FROM support_tickets WHERE customer_name = 'Denise Lee'\nResult: [{\"customer_name\": \"Denise Lee\"}]"
    sql, result = _parse_sql_tool_output(out)
    assert sql == "SELECT * FROM support_tickets WHERE customer_name = 'Denise Lee'"
    assert "Denise Lee" in (result or "")


def test_parse_sql_tool_output_no_matching():
    """Parse tool output when no matching data."""
    out = "SQL: SELECT * FROM support_tickets WHERE customer_name = 'X'\nResult: No matching data found."
    sql, result = _parse_sql_tool_output(out)
    assert sql is not None
    assert "No matching data found" in (result or "")


def test_agent_response_dataclass():
    """AgentResponse has required fields."""
    r = AgentResponse(
        answer="Test answer",
        sql_query="SELECT 1",
        sql_result="[]",
        retrieval_used=False,
        internal_query="test",
        nlp_details={"intent": "customer"},
        agent_selection="query_customer_tickets",
    )
    assert r.answer == "Test answer"
    assert r.sql_query == "SELECT 1"
    assert r.nlp_details["intent"] == "customer"
