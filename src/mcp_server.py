"""MCP server: exposes agent so UI or CLI can call it. Run with: python -m src.mcp_server."""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def chat(message: str) -> str:
    """Single entry point: run agent and return JSON with answer and optional query details."""
    from src.agent.agent import invoke
    r = invoke(message)
    return json.dumps({
        "answer": r.answer,
        "sql_query": r.sql_query,
        "sql_result": r.sql_result,
        "retrieval_used": r.retrieval_used,
        "retrieval_snippet": r.retrieval_snippet,
    })


def main_sync():
    """Read one line from stdin, call chat, print JSON to stdout."""
    try:
        line = sys.stdin.readline()
        if not line:
            return
        msg = line.strip()
        out = chat(msg)
        print(out)
    except Exception as e:
        print(json.dumps({"answer": f"Error: {e}", "sql_query": None, "sql_result": None, "retrieval_used": False, "retrieval_snippet": None}))


if __name__ == "__main__":
    main_sync()
