"""
rag/retriever.py
----------------
PIPELINE STEP 3 — Find the most relevant chunks for a user's question.

HOW SIMILARITY SEARCH WORKS:
  1. The user types: "What caused the database outage?"
  2. We embed that question using the same model (text-embedding-3-small)
     → question_vector = [0.14, -0.47, 0.85, ...]
  3. FAISS computes cosine similarity between question_vector and every
     stored chunk vector
  4. Returns the TOP_K chunks with highest similarity scores

  The key insight: chunks that *talk about the same topic* as the question
  will have high cosine similarity, even if they use different words.
  "DB timeout" and "database connection refused" are semantically close.

WHY NOT SEND ALL CHUNKS TO THE LLM?
  A 1000-line log file → ~50 chunks. Sending all 50 to GPT-4 would:
    - Use ~10,000 tokens → cost more
    - Overwhelm the model with irrelevant context
    - Slow down the response

  By retrieving only the TOP_K=5 most relevant chunks, we give the LLM
  exactly what it needs to answer the specific question.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import TOP_K


def retrieve_relevant_chunks(
    vectorstore: FAISS,
    query: str,
    k: int = TOP_K,
) -> list[Document]:
    """
    Find the K most relevant chunks for the given query.

    Args:
        vectorstore: The FAISS index built by embedder.py
        query:       The user's question (e.g. "What caused the DB outage?")
        k:           Number of chunks to return (default from settings.TOP_K)

    Returns:
        List of Document objects, sorted by relevance (most relevant first).
        Each Document has:
          - page_content : the raw log lines in this chunk
          - metadata     : { start_line, end_line, error_count, warning_count }
    """
    # similarity_search() internally:
    #   1. Embeds `query` using the same embedding model
    #   2. Computes cosine similarity against all stored vectors
    #   3. Returns the k nearest neighbors
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def retrieve_with_scores(
    vectorstore: FAISS,
    query: str,
    k: int = TOP_K,
) -> list[tuple[Document, float]]:
    """
    Same as retrieve_relevant_chunks() but also returns similarity scores.
    Scores range from 0.0 (unrelated) to 1.0 (identical).
    Useful for debugging — shown in the Streamlit UI's source expander.
    """
    return vectorstore.similarity_search_with_score(query, k=k)
