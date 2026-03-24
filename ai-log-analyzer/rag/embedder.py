"""
rag/embedder.py
---------------
PIPELINE STEP 2 — Convert chunks into vectors and store in FAISS.

WHAT IS AN EMBEDDING?
  An embedding is a list of floating-point numbers that represents
  the *meaning* of a piece of text. Texts with similar meanings end up
  close together in this high-dimensional space.

  Example:
    "database connection timeout"  →  [0.12, -0.45, 0.88, ...]
    "DB refused to connect"        →  [0.11, -0.43, 0.91, ...]  ← similar!
    "user logged in successfully"  →  [-0.72, 0.34, -0.22, ...]  ← different

  We use a local sentence-transformers model (all-MiniLM-L6-v2) to generate
  these vectors. It runs on your machine — no API key needed, completely free.

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is an in-memory vector database.
  It stores all chunk embeddings and supports fast nearest-neighbor search:
  "Given a query vector, find the K most similar chunk vectors."

  We use FAISS because:
    - No server to set up (runs in-process, in memory)
    - Extremely fast even on large datasets
    - LangChain has built-in FAISS integration

HOW IT WORKS:
  1. We embed all chunk texts using the local sentence-transformers model
  2. Each chunk becomes a 384-dim float vector
  3. We store those vectors in a FAISS index
  4. Later, when the user asks a question, the same embedding model
     converts the question to a vector, and FAISS finds the closest chunks
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import EMBEDDING_MODEL


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embed all chunks and store them in a FAISS vector index.

    Args:
        chunks: List of Document objects from chunker.py

    Returns:
        A FAISS vectorstore ready for similarity search.

    Note:
        Uses all-MiniLM-L6-v2 (local, free, no API key required).
        First run downloads the model (~90 MB) from HuggingFace.
    """
    if not chunks:
        raise ValueError("No chunks provided — cannot build an empty vector store.")

    # HuggingFaceEmbeddings runs the model locally via sentence-transformers.
    # The model is downloaded on first use and cached in ~/.cache/huggingface/
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # FAISS.from_documents():
    #   1. Calls embeddings.embed_documents([chunk.page_content for chunk in chunks])
    #   2. Returns a list of 384-dim vectors
    #   3. Builds an in-memory FAISS index
    #   4. Stores both the vectors AND the original Document objects
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
