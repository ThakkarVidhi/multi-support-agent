"""ReAct-style agent with SQL and Retriever tools. Returns structured response for UI toggle."""
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured response for UI: answer + optional query details."""
    answer: str
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    retrieval_used: bool = False
    retrieval_snippet: Optional[str] = None


SYSTEM_PROMPT = """You are a support assistant. You have two tools: one for querying customer and support ticket data (SQL), and one for searching policy documents. Use the SQL tool for questions about specific customers or tickets; use the retriever for policy/refund/terms. If a question needs both (e.g. does customer X qualify under the refund policy?), call both tools then summarize. Always answer in clear, concise language."""


def _parse_sql_tool_output(output: str) -> tuple[Optional[str], Optional[str]]:
    """Parse 'SQL: ... \\nResult: ...' from tool output."""
    sql_query = None
    sql_result = None
    if output.startswith("SQL:"):
        parts = output.split("\nResult:", 1)
        if len(parts) >= 1:
            sql_query = parts[0].replace("SQL:", "").strip()
        if len(parts) >= 2:
            sql_result = parts[1].strip()
    return sql_query, sql_result


def build_agent():
    """Build agent: Ollama LLM + SQL tool + Retriever tool."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, DB_PATH, CHROMA_PATH
    from src.agent.tools import create_sql_tool, create_retriever_tool

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)
    sql_tool = create_sql_tool(llm=llm, db_path=DB_PATH)
    retriever_tool = create_retriever_tool(chroma_path=CHROMA_PATH)
    tools = [sql_tool, retriever_tool]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools, llm


def invoke(message: str) -> AgentResponse:
    """Run agent on user message. Return structured response (answer + sql_query, sql_result, retrieval_*)."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

    llm_with_tools, tools, llm = build_agent()
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    retrieval_used = False
    retrieval_snippet: Optional[str] = None
    max_iterations = 5

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        if not getattr(response, "tool_calls", None):
            answer = response.content if hasattr(response, "content") else str(response)
            return AgentResponse(
                answer=answer or "I couldn't generate a response.",
                sql_query=sql_query,
                sql_result=sql_result,
                retrieval_used=retrieval_used,
                retrieval_snippet=retrieval_snippet[:2000] if retrieval_snippet else None,
            )
        messages.append(response)
        for tc in response.tool_calls:
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
            tid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            if name not in tool_map:
                continue
            tool = tool_map[name]
            try:
                output = tool.invoke(args)
            except Exception as e:
                output = f"Error: {e}"
            if name == "query_customer_tickets":
                sq, sr = _parse_sql_tool_output(str(output))
                if sq is not None:
                    sql_query = sq
                if sr is not None:
                    sql_result = sr
            elif name == "search_policy_documents":
                retrieval_used = True
                retrieval_snippet = str(output)[:2000] if output else None
            messages.append(ToolMessage(content=str(output), tool_call_id=tid))

    answer = "I couldn't complete the request. Please try again."
    return AgentResponse(
        answer=answer,
        sql_query=sql_query,
        sql_result=sql_result,
        retrieval_used=retrieval_used,
        retrieval_snippet=retrieval_snippet[:2000] if retrieval_snippet else None,
    )
