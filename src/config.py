"""Central config: env vars and paths. No secrets in code."""
import os
from pathlib import Path

# Project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
def _load_dotenv() -> None:
    try:
        import dotenv
        dotenv.load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

_load_dotenv()

# Ollama (local model only)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Paths (relative to project root or absolute)
_db_path = os.getenv("DB_PATH", "data/customer_support.db")
_chroma_path = os.getenv("CHROMA_PATH", "data/chroma_policies")

DB_PATH = Path(_db_path) if os.path.isabs(_db_path) else PROJECT_ROOT / _db_path
CHROMA_PATH = Path(_chroma_path) if os.path.isabs(_chroma_path) else PROJECT_ROOT / _chroma_path
