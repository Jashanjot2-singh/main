import os
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    from langchain_ollama import OllamaEmbeddings  # type: ignore


COLLECTION_NAME = "abc_company_docs"
VECTOR_SIZE = 768
DISTANCE = Distance.COSINE

# Your PDFs (one at a time in a loop)
PDF_FILES = ["docs_1.pdf", "docs_2.pdf", "docs_3.pdf", "docs_4.pdf", "docs_5.pdf"]

# Chunking requested
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50  # will be auto-adjusted to 49 if needed

# Ollama embedding model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Qdrant Cloud creds (use env vars; don’t hardcode secrets in git)
QDRANT_URL = os.getenv("QDRANT_URL", "https://85ea9aea-a625-4e28-8e3e-47db7073a86c.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", )


def safe_overlap(chunk_size: int, overlap: int) -> int:
    if overlap >= chunk_size:
        print(f"[WARN] chunk_overlap ({overlap}) >= chunk_size ({chunk_size}). "
              f"Adjusting overlap to {chunk_size - 1} to avoid splitter errors.")
        return max(0, chunk_size - 1)
    return overlap


def ensure_collection(qdrant: QdrantClient, name: str) -> None:
    """
    Create collection if it doesn't exist.
    """
    existing = {c.name for c in qdrant.get_collections().collections}
    if name in existing:
        return

    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
    )
    print(f"[OK] Created Qdrant collection: {name}")


def load_and_chunk(pdf_path: str, doc_name: str):
    """
    Load a single PDF and split into chunks. Adds doc_name into metadata for every chunk.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # list[Document]; each has page_content + metadata (page, source, etc.)

    overlap = safe_overlap(CHUNK_SIZE, CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  # good general-purpose defaults
    )

    chunks = splitter.split_documents(pages)

    # Add/normalize metadata
    for i, d in enumerate(chunks):
        d.metadata = d.metadata or {}
        d.metadata["doc_name"] = doc_name           # e.g., "doc1"
        d.metadata["chunk_id"] = i                  # chunk index within doc
        # keep d.metadata["page"] if present from PyPDFLoader
    return chunks


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    if not QDRANT_API_KEY:
        print("[WARN] QDRANT_API_KEY env var is empty. If your Qdrant requires auth, set it.")
    if "<waiting-for-cluster-host>" in QDRANT_URL:
        print("[WARN] Update QDRANT_URL to your real Qdrant Cloud endpoint.")

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(qdrant.get_collections())

    ensure_collection(qdrant, COLLECTION_NAME)

    embedder = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    for idx, pdf in enumerate(PDF_FILES, start=1):
        if not os.path.exists(pdf):
            print(f"[SKIP] {pdf} not found. Put it in this folder or edit PDF_FILES.")
            continue

        doc_tag = f"doc{idx}"  # metadata name requested: doc1, doc2, ...
        print(f"\n[INGEST] {pdf} as {doc_tag}")

        chunks = load_and_chunk(pdf, doc_tag)
        texts: List[str] = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]

        # Embed in batches (safer for large docs)
        all_points = []
        BATCH = 64

        for b_start in range(0, len(texts), BATCH):
            b_texts = texts[b_start:b_start + BATCH]
            b_metas = metas[b_start:b_start + BATCH]

            vectors = embedder.embed_documents(b_texts)

            # Validate embedding dimension once
            if vectors and len(vectors[0]) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding dim mismatch: got {len(vectors[0])}, expected {VECTOR_SIZE}. "
                    f"Check your embedding model."
                )

            for text, meta, vec in zip(b_texts, b_metas, vectors):
                payload = dict(meta)
                payload["text"] = text  # store chunk text directly in Qdrant payload

                all_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload=payload,
                    )
                )

        # Upsert to Qdrant
        for batch_points in batched(all_points, 256):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=batch_points,
            )

        print(f"[OK] Upserted {len(all_points)} chunks for {doc_tag}")

    print("\n[DONE] Ingestion complete.")


if __name__ == "__main__":
    main()
