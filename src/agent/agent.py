"""ReAct-style agent with intent-based routing. Policy -> retriever only; Customer -> SQL only; Both -> both."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agent.intent import (
    INTENT_BOTH,
    INTENT_CUSTOMER,
    INTENT_POLICY,
    classify_intent_and_entities,
    IntentResult,
)

logger = logging.getLogger(__name__)

CHECKPOINT = "checkpoint"

def _log_checkpoint(name: str, data: Dict[str, Any]) -> None:
    logger.info("[%s] %s: %s", CHECKPOINT, name, data)


@dataclass
class AgentResponse:
    """Structured response for UI: answer + query details for expandable panel."""
    answer: str
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    retrieval_used: bool = False
    retrieval_snippet: Optional[str] = None
    internal_query: Optional[str] = None
    nlp_details: Optional[Dict[str, Any]] = None 
    agent_selection: Optional[str] = None 
    raw_tool_output: Optional[str] = None 


FINAL_ANSWER_SYSTEM = """You are a support assistant. Answer the user's question using ONLY the context below.

Formatting rules:
- Formatting (strict — use line breaks for readability):
- When presenting customer or ticket data: use natural language and a clear structure. For example:
    - **Profile** section: Put each item on its own line. Example:
        **Profile**
            - Name: [name]
            - Email: [email]
            - Age: [age], Gender: [gender]
            - Product: [product_purchased]
            - Date of purchase: [date]
    - **Tickets** section: One ticket per block, each field on its own line. Example:
        **Tickets**
            - **Ticket #[id]** — [ticket_type]: [ticket_subject]
            - Status: [ticket_status]
            - Priority: [ticket_priority]
            - Description: [one short sentence]
            - Resolution: [resolution or "None yet"]
            - Use a blank line between Profile and Tickets. No long run-on lines; use bullets and newlines.
- For policy-only: answer in clear natural language with short paragraphs.

Content rules:
- If the context contains customer/ticket data (e.g. JSON rows), present it in the structured natural-language format above. Do NOT say "no matching data found" when the context clearly contains data.
- Only say "no matching data found" when the context explicitly says "No matching data found" or the result is empty.
- For "Does [customer] qualify under the refund policy?": combine (1) what we found about the customer (tickets, purchase type) and (2) what the policy says about refunds, then state whether they qualify and why in 2–4 sentences.
- Do not invent data. Be concise and clear."""


def _parse_sql_tool_output(output: str) -> tuple[Optional[str], Optional[str]]:
    """Parse 'SQL: ... \\nResult: ...' from tool output."""
    sql_query = None
    sql_result = None
    if "SQL:" in output and "Result:" in output:
        parts = output.split("\nResult:", 1)
        if len(parts) >= 1:
            sql_query = parts[0].replace("SQL:", "").strip()
        if len(parts) >= 2:
            sql_result = parts[1].strip()
    return sql_query, sql_result


def build_agent():
    """Build LLM and tools (for intent + tool execution)."""
    from langchain_ollama import ChatOllama
    from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, DB_PATH, CHROMA_PATH
    from src.agent.tools import create_sql_tool, create_retriever_tool

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)
    sql_tool = create_sql_tool(llm=llm, db_path=DB_PATH)
    retriever_tool = create_retriever_tool(chroma_path=CHROMA_PATH)
    tools = [sql_tool, retriever_tool]
    return llm, tools, {t.name: t for t in tools}


def invoke(message: str) -> AgentResponse:
    """
    Run intent classification, route to correct tool(s) only, then generate final answer.
    No policy questions to SQL; no customer-only questions to retriever.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    raw_query = (message or "").strip()
    _log_checkpoint("User Input Received", {"message_length": len(raw_query)})
    _log_checkpoint("Raw User Query", {"query": raw_query})

    if not raw_query:
        return AgentResponse(
            answer="Please provide a question about a customer, support tickets, or company policy.",
            internal_query=raw_query,
            nlp_details={"intent": "none", "confidence": 0, "entities": {}},
            agent_selection="none",
        )

    llm, tools, tool_map = build_agent()
    intent_result: IntentResult = classify_intent_and_entities(raw_query, llm=llm)

    _log_checkpoint("Intent Classification Result", {
        "selected_intent": intent_result.intent,
        "confidence_score": intent_result.confidence,
    })
    
    entities = dict(intent_result.entities) if getattr(intent_result, "entities", None) else {}
    if intent_result.customer_name and "customer_name" not in entities:
        entities["customer_name"] = intent_result.customer_name
    if intent_result.ticket_id and "ticket_id" not in entities:
        entities["ticket_id"] = intent_result.ticket_id
    _log_checkpoint("Entity Extraction", {"extracted_entities": entities})

    # Route: only call the appropriate tool(s)
    selected_tools = (
        ["query_customer_tickets", "search_policy_documents"] if intent_result.intent == INTENT_BOTH else
        ["query_customer_tickets"] if intent_result.intent == INTENT_CUSTOMER else
        ["search_policy_documents"]
    )
    _log_checkpoint("Agent Selection", {
        "selected_agent": selected_tools[0] if len(selected_tools) == 1 else "query_customer_tickets, search_policy_documents",
        "reason": "Intent-based routing: " + intent_result.intent,
    })

    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    retrieval_used = False
    retrieval_snippet: Optional[str] = None
    tool_outputs: List[str] = []

    if intent_result.intent in (INTENT_CUSTOMER, INTENT_BOTH):
        try:
            out = tool_map["query_customer_tickets"].invoke({"question": raw_query})
            tool_outputs.append("SQL tool: " + str(out)[:500])
            sq, sr = _parse_sql_tool_output(str(out))
            if sq is not None:
                sql_query = sq
            if sr is not None:
                sql_result = sr
            _log_checkpoint("Query Generation", {"final_query_tool": "query_customer_tickets", "sql": sql_query})
        except Exception as e:
            tool_outputs.append("SQL tool error: " + str(e))
            sql_result = "Error: " + str(e)
            _log_checkpoint("Query Generation", {"error": str(e)})

    if intent_result.intent in (INTENT_POLICY, INTENT_BOTH):
        try:
            out = tool_map["search_policy_documents"].invoke({"query": raw_query})
            retrieval_used = True
            retrieval_snippet = str(out)[:2000] if out else None
            tool_outputs.append("Retriever: " + (str(out)[:500] if out else "No results"))
            _log_checkpoint("Query Generation", {"final_query_tool": "search_policy_documents"})
        except Exception as e:
            tool_outputs.append("Retriever error: " + str(e))
            retrieval_snippet = "Error: " + str(e)

    # Build context for final answer
    context_parts = []
    if sql_result is not None:
        context_parts.append("Customer/ticket data:\n" + sql_result)
    if retrieval_snippet:
        context_parts.append("Policy/relevant documents:\n" + retrieval_snippet)
    if not context_parts:
        context_parts.append("No data was retrieved. Tell the user no matching data was found and do not invent any information.")

    context = "\n\n".join(context_parts)
    messages = [
        SystemMessage(content=FINAL_ANSWER_SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nUser question: {raw_query}\n\nAnswer in natural language with clear structure. For customer/ticket data, use Profile and Tickets sections as in the instructions. For policy questions, answer in plain language. For 'qualify under refund policy', combine customer data and policy and state whether they qualify."),
    ]
    try:
        response = llm.invoke(messages)
        raw_answer = response.content if hasattr(response, "content") else str(response)
        _log_checkpoint("Agent Response", {"raw_output": raw_answer[:500] if raw_answer else ""})
    except Exception as e:
        raw_answer = f"I couldn't complete the request: {e}"
        logger.exception("LLM final answer failed")
        _log_checkpoint("Agent Response", {"error": str(e)})

    nlp_details = {
        "intent": intent_result.intent,
        "confidence": intent_result.confidence,
        "entities": entities,
    }
    agent_selection = ", ".join(selected_tools)
    payload = {
        "answer": raw_answer,
        "internal_query": raw_query,
        "nlp_details": nlp_details,
        "agent_selection": agent_selection,
    }
    _log_checkpoint("UI Payload", {"answer_length": len(raw_answer), "has_sql": sql_query is not None, "has_retrieval": retrieval_used})

    return AgentResponse(
        answer=raw_answer or "I couldn't generate a response.",
        sql_query=sql_query,
        sql_result=sql_result,
        retrieval_used=retrieval_used,
        retrieval_snippet=retrieval_snippet[:2000] if retrieval_snippet else None,
        internal_query=raw_query,
        nlp_details=nlp_details,
        agent_selection=agent_selection,
        raw_tool_output=" | ".join(tool_outputs) if tool_outputs else None,
    )
