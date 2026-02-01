"""LangChain tools: SQL (query_customer_tickets) and Retriever (search_policy_documents)."""
import logging
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


SQL_SYSTEM_PROMPT = """You are an expert SQLite query generator. Generate a single SELECT for the support_tickets table using the schema below.

Schema:
{table_info}

Guidelines:
- Use ONLY the column names from the schema. Do not invent column names.
- Filter using whichever columns match the user's question: customer name -> customer_name (LIKE '%name%'); email -> customer_email; product -> product_purchased (LIKE '%product%'); ticket status -> ticket_status; ticket ID -> ticket_id.
- Use LIKE with wildcards for text when the user may give partial info (e.g. customer_name LIKE '%Denise%' AND customer_name LIKE '%Lee%', or product_purchased LIKE '%Philips%').
- Combine multiple conditions with AND when the user mentions name + product, or email + status, etc.
- For "overview of customer X" or "tickets for X", return all columns for that customer (WHERE customer_name LIKE '%X%' or similar).
- For "Does [customer] qualify under the refund policy?" or similar: return ALL tickets for that customer (WHERE customer_name LIKE '%name%'). Do NOT add AND ticket_status = 'Refunded' or filter by status; we need their tickets (e.g. Refund request, Open) to decide eligibility.
- Return ONLY the SQL SELECT, no explanation or markdown."""


def create_sql_tool(llm=None, db_path=None):
    """Create SQL tool: NL -> SELECT -> execute -> result string.
    Use when user asks about customer, tickets, support history.
    """
    from src.db import sqlite_client

    schema = sqlite_client.get_schema(db_path)

    @tool
    def query_customer_tickets(question: str) -> str:
        """Use this when the user asks about a specific customer, support tickets, ticket history, customer profile, or any structured data about customers or tickets. Input: the user's question or a rewritten question focused on customer/ticket data."""
        if not question or not question.strip():
            return "Please provide a question about customers or support tickets."
        try:
            if llm is None:
                from langchain_ollama import ChatOllama
                from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
                _llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)
            else:
                _llm = llm
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", SQL_SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            chain = prompt | _llm
            response = chain.invoke({"table_info": schema, "question": question})
            sql = response.content if hasattr(response, "content") else str(response)
            sql = sql.strip()
            if "```" in sql:
                for part in sql.split("```"):
                    if "select" in part.lower():
                        sql = part.strip()
                        break
            if not sql.upper().startswith("SELECT"):
                return "Generated query was not a SELECT. Please ask about customer or ticket data."
            rows = sqlite_client.run_query(sql, db_path)
            if rows and len(rows) == 1 and "error" in rows[0]:
                return "SQL: " + sql + "\nResult: Query error: " + str(rows[0]["error"])
            if not rows:
                return "SQL: " + sql + "\nResult: No matching data found."
            import json
            result_str = json.dumps(rows, default=str)[:8000]
            return "SQL: " + sql + "\nResult: " + result_str
        except Exception as e:
            logger.exception("SQL tool error")
            return f"Error: {str(e)}"

    return query_customer_tickets


def create_retriever_tool(chroma_path=None):
    """Create retriever tool: query -> embed -> Chroma search -> concatenated chunks.
    Use when user asks about policy, refund, terms.
    """
    from src.db import vector_store

    @tool
    def search_policy_documents(query: str) -> str:
        """Use this when the user asks about company policy, refund policy, terms, cancellation, or any information that would be in policy or legal documents. Input: the user's question or a rewritten question focused on policy/refund/terms."""
        if not query or not query.strip():
            return "Please provide a question about policy or documents."
        try:
            text = vector_store.search(query, k=3, chroma_path=chroma_path)
            return text if text else "No relevant policy documents found."
        except Exception as e:
            logger.exception("Retriever tool error")
            return f"Error: {str(e)}"

    return search_policy_documents
