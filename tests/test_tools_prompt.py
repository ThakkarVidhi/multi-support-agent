"""Tests that SQL prompt includes qualify-under-policy rule (no Refunded filter)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.agent.tools import SQL_SYSTEM_PROMPT


def test_sql_prompt_qualify_under_policy_rule():
    """SQL prompt tells LLM not to filter by Refunded for qualify-under-policy questions."""
    prompt_lower = SQL_SYSTEM_PROMPT.lower()
    assert "qualify" in prompt_lower
    assert "refunded" in prompt_lower
    assert "do not" in prompt_lower
    assert "ticket_status" in prompt_lower
