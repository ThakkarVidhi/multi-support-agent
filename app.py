"""Streamlit UI: chat with toggle for query details. Calls agent directly."""
import sys
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.agent.agent import invoke, AgentResponse

st.set_page_config(page_title="Data Loom", page_icon="🪢", layout="centered")
st.title("Data Loom")
st.caption("Ask about customer tickets or company policy (e.g. refund policy).")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# Toggle: Show query details
st.session_state.show_details = st.sidebar.checkbox(
    "Show query details (SQL query, result, policy snippet)",
    value=st.session_state.show_details,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            d = msg["content"]
            if st.session_state.show_details:
                if d.get("sql_query"):
                    st.markdown("**SQL query**")
                    st.code(d["sql_query"], language="sql")
                if d.get("sql_result") is not None:
                    st.markdown("**SQL result**")
                    st.text(d["sql_result"][:2000] + ("..." if len(d.get("sql_result") or "") > 2000 else ""))
                if d.get("retrieval_used") and d.get("retrieval_snippet"):
                    st.markdown("**Policy snippet**")
                    st.text(d["retrieval_snippet"][:1500] + ("..." if len(d.get("retrieval_snippet") or "") > 1500 else ""))
            st.markdown(d.get("answer", ""))
        else:
            st.markdown(msg.get("content", ""))

if prompt := st.chat_input("Ask about a customer or policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp: AgentResponse = invoke(prompt)
            except Exception as e:
                resp = AgentResponse(answer=f"The assistant is temporarily unavailable: {e}")
        d = {
            "answer": resp.answer,
            "sql_query": resp.sql_query,
            "sql_result": resp.sql_result,
            "retrieval_used": resp.retrieval_used,
            "retrieval_snippet": resp.retrieval_snippet,
        }
        st.session_state.messages.append({"role": "assistant", "content": d})
        if st.session_state.show_details:
            if d.get("sql_query"):
                st.markdown("**SQL query**")
                st.code(d["sql_query"], language="sql")
            if d.get("sql_result") is not None:
                st.markdown("**SQL result**")
                st.text(d["sql_result"][:2000] + ("..." if len(d.get("sql_result") or "") > 2000 else ""))
            if d.get("retrieval_used") and d.get("retrieval_snippet"):
                st.markdown("**Policy snippet**")
                st.text(d["retrieval_snippet"][:1500] + ("..." if len(d.get("retrieval_snippet") or "") > 1500 else ""))
        st.markdown(d["answer"])
