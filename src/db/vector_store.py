"""Chroma vector store: ingest PDFs, similarity search. Uses sentence-transformers all-MiniLM-L6-v2."""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Lazy init
_client = None
_embedding_fn = None


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        from sentence_transformers import SentenceTransformer
        _embedding_fn = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_fn


def _embed(texts: List[str]) -> List[List[float]]:
    model = _get_embedding_fn()
    return model.encode(texts).tolist()


def get_client(chroma_path: Path | None = None):
    """Get or create Chroma persistent client."""
    global _client
    if _client is None:
        from src.config import CHROMA_PATH
        import chromadb
        path = str(chroma_path or CHROMA_PATH)
        Path(path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)
    return _client


def ingest_pdfs(directory: Path | str, chroma_path: Path | None = None) -> None:
    """Load PDFs from directory, chunk, embed, add to Chroma collection 'policy_docs'.
    Uses sentence-transformers all-MiniLM-L6-v2 (same as search). Re-run clears and re-adds."""
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    directory = Path(directory)
    if not directory.exists():
        logger.warning("Directory %s does not exist", directory)
        return

    client = get_client(chroma_path)
    try:
        client.delete_collection(name="policy_docs")
    except Exception:
        pass
    collection = client.get_or_create_collection(name="policy_docs", metadata={"description": "Policy PDF chunks"})

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "])
    all_ids = []
    all_docs = []
    all_metadatas = []

    for pdf_path in directory.glob("*.pdf"):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            chunks = splitter.split_text(text)
            if not chunks:
                continue
            ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
            all_ids.extend(ids)
            all_docs.extend(chunks)
            all_metadatas.extend([{"source": pdf_path.name}] * len(chunks))
        except Exception as e:
            logger.exception("Failed to ingest %s: %s", pdf_path, e)

    if not all_docs:
        logger.warning("No chunks to add")
        return

    embeddings = _embed(all_docs)

    # Chroma add in batches
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        end = min(i + batch_size, len(all_docs))
        collection.add(
            ids=all_ids[i:end],
            documents=all_docs[i:end],
            embeddings=embeddings[i:end],
            metadatas=all_metadatas[i:end],
        )
    logger.info("Ingested %d chunks from %s", len(all_docs), directory)


def search(query: str, k: int = 3, chroma_path: Path | None = None) -> str:
    """Embed query, run similarity search, return concatenated chunk text. k=3 for lower latency."""
    client = get_client(chroma_path)
    try:
        collection = client.get_collection(name="policy_docs")
    except Exception:
        return ""

    query_embedding = _embed([query])
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    if not results or not results.get("documents"):
        return ""
    docs = results["documents"][0]
    return "\n\n".join(docs) if docs else ""


def warmup(chroma_path: Path | None = None) -> None:
    """Load embedding model and Chroma client so the first policy query is fast. Call at app startup."""
    try:
        _get_embedding_fn()
        client = get_client(chroma_path)
        try:
            client.get_collection(name="policy_docs")
        except Exception:
            pass
    except Exception as e:
        logger.debug("Vector store warmup skipped or failed: %s", e)
