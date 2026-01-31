"""Run vector_store.ingest_pdfs on data/policies."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.vector_store import ingest_pdfs
from src.config import CHROMA_PATH

def main() -> None:
    policies_dir = PROJECT_ROOT / "data" / "policies"
    ingest_pdfs(policies_dir)
    print(f"Ingested PDFs from {policies_dir} -> Chroma at {CHROMA_PATH}")

if __name__ == "__main__":
    main()
