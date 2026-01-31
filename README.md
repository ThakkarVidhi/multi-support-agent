# DataLoom — Multi-Agent Support Assistant

**A Generative AI Multi-Agent system that answers support questions using natural language over SQL (customer/ticket data) and policy documents (RAG).**

---

## Project Description

This project is a **Streamlit** chat application powered by a **LangChain** agent with two tools: (1) **SQL tool** — queries a SQLite database of customer support tickets (Kaggle-style data); (2) **Retriever tool** — searches policy PDFs in a **Chroma** vector store. The LLM runs **locally via Ollama** (Llama 3.2 3B), so no API keys are required. The UI includes an optional **"Show query details"** toggle to display the generated SQL query, SQL result, and policy snippet before the final answer.

---

## Features

- **Natural language to SQL**: Ask about customers or tickets (e.g. "Give me a quick overview of customer Ema's profile and past support ticket details") — the agent generates and runs a SELECT query.
- **Policy RAG**: Ask about refund policy or terms (e.g. "What is the current refund policy?") — the agent retrieves relevant chunks from pre-loaded policy PDFs and summarizes.
- **Local LLM (Ollama)**: Uses Llama 3.2 3B via Ollama — no API keys; runs fully offline after setup.
- **Streamlit UI**: Single-page chat with message history and optional query-details toggle.
- **MCP server**: Optional script (`python -m src.mcp_server`) to call the agent via stdin/stdout.

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python, LangChain, LangGraph (agent + tools) |
| **LLM** | Ollama (Llama 3.2 3B) — local |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **SQL DB** | SQLite (one table: support_tickets) |
| **Vector DB** | Chroma (policy PDF chunks) |
| **Config** | python-dotenv, .env |

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (`ollama pull llama3.2:3b`)
- **Kaggle account** (optional) — for the official CSV; or use the included synthetic CSV.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd tts-assignment
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Environment**
   - Copy `.env.example` to `.env` and set:
     - `OLLAMA_BASE_URL=http://localhost:11434` (default)
     - `OLLAMA_MODEL=llama3.2:3b`
     - `DB_PATH=data/customer_support.db`
     - `CHROMA_PATH=data/chroma_policies`

4. **Data**
   - **Policy PDF**: Run `python scripts/download_policy_pdf.py` to download the refund policy PDF to `data/policies/`.
   - **CSV**: Either place the Kaggle "Customer Support Ticket Dataset" CSV in `data/raw/`, or use the included `data/raw/synthetic_support_tickets.csv` (includes customer "Ema" for the demo).

5. **Seed SQLite and ingest policies**
   ```bash
   python data/seed_db.py
   python scripts/ingest_policies.py
   ```
   (First run of `ingest_policies.py` may take a few minutes while the embedding model downloads.)

---

## Usage

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. Open the app in your browser (e.g. http://localhost:8501).

4. **Example questions**
   - "What is the current refund policy?"
   - "Give me a quick overview of customer Ema's profile and past support ticket details."

5. Use the sidebar checkbox **"Show query details"** to see the SQL query, SQL result, and policy snippet (if used) before the final answer.

---

## Configuration

See `.env.example`. No API keys are required when using the local Ollama model.

---

## Testing

1. Launch the app with `streamlit run app.py`.
2. Ask: "What is the current refund policy?" — you should get a summary from the policy PDF.
3. Ask: "Give me a quick overview of customer Ema's profile and past support ticket details." — you should get data from the SQLite table.
4. Toggle "Show query details" on and repeat — you should see the SQL query and result (for the second question) and policy snippet (for the first).

---

## Architecture

- **UI**: Streamlit → calls `agent.invoke(message)`.
- **Agent**: ReAct-style loop with two tools (SQL, Retriever); LLM (Ollama) decides which tool(s) to call and produces the final answer.
- **SQL tool**: NL → LLM generates SELECT → SQLite → result string.
- **Retriever tool**: Query → embed → Chroma similarity search → concatenated chunks.
- **Data**: SQLite (one table); Chroma (policy chunks).

---

## Demo Video

[Add your demo video URL here after recording.]

---

## License

MIT.

---

## Contact

[Your contact information.]
