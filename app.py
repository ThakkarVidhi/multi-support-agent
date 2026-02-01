import logging
import sys
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Checkpoint logging at INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import streamlit as st
from src.agent.agent import invoke, AgentResponse

# Warmup RAG (embedding model + Chroma) policy question is fast
try:
    from src.db import vector_store
    vector_store.warmup()
except Exception:
    pass

st.set_page_config(page_title="Data Loom", page_icon="🪢", layout="centered")

st.markdown("""
<style>
  .stChatMessage { max-width: 100%; }
  [data-testid="stChatMessage"] { align-self: stretch; }
  .title-with-logo { display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="title-with-logo"><span style="font-size:2rem">🪢</span> <span style="font-size:2rem; font-weight:600">Data Loom</span></p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:gray; margin-top:-0.5rem;">Ask about customer tickets or company policy (e.g. refund policy).</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

def _render_query_details(d: dict) -> None:
    """Single place to render query details expander content (avoids duplication)."""
    with st.expander("Show query details", expanded=False):
        st.markdown("**Internal query**")
        st.text(d.get("internal_query") or "(same as user message)")
        st.markdown("**NLP analysis details**")
        nlp = d.get("nlp_details") or {}
        st.markdown(f"- **Intent:** `{nlp.get('intent', '—')}`")
        st.markdown(f"- **Confidence:** `{nlp.get('confidence', '—')}`")
        st.markdown("- **Entities:**")
        entities = nlp.get("entities") or {}
        for k, v in entities.items():
            st.text(f"  {k}: {v}")
        st.markdown("**Agent selection**")
        st.text(d.get("agent_selection") or "—")
        st.markdown("**Backend tool results**")
        if d.get("sql_query"):
            st.markdown("*SQL query*")
            st.code(d["sql_query"], language="sql")
        if d.get("sql_result") is not None:
            st.markdown("*SQL result*")
            st.text(d["sql_result"][:3000] + ("..." if len(d.get("sql_result") or "") > 3000 else ""))
        if d.get("retrieval_used") and d.get("retrieval_snippet"):
            st.markdown("*Policy snippet*")
            st.text(d["retrieval_snippet"][:2000] + ("..." if len(d.get("retrieval_snippet") or "") > 2000 else ""))
        if not d.get("sql_query") and not d.get("retrieval_used"):
            st.text("(No tool results for this message)")


def _message_container(i: int):
    """Keyed container if supported (avoids duplicate expanders); else no-op context."""
    try:
        return st.container(key=f"msg_{i}")
    except TypeError:
        from contextlib import nullcontext
        return nullcontext()


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        with _message_container(i):
            if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
                d = msg["content"]
                st.markdown(d.get("answer", ""))
                _render_query_details(d)
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
            "internal_query": resp.internal_query,
            "nlp_details": resp.nlp_details,
            "agent_selection": resp.agent_selection,
            "raw_tool_output": resp.raw_tool_output,
        }
        st.session_state.messages.append({"role": "assistant", "content": d})
        with _message_container(len(st.session_state.messages) - 1):
            st.markdown(d["answer"])
            _render_query_details(d)
