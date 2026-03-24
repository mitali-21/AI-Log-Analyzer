"""
rag/chunker.py
--------------
PIPELINE STEP 1 — Split the log text into overlapping chunks.

WHY CHUNKING?
  A log file can have thousands of lines. Sending everything to the LLM
  at once would:
    - Exceed the context window limit
    - Cost more tokens
    - Dilute the relevant content with noise

  Instead, we split into small, overlapping windows of log lines.
  Each chunk becomes a searchable unit in the vector database.

WHY OVERLAP?
  If a multi-line error spans lines 19-21 and our chunk boundary falls
  at line 20, without overlap we'd cut the error in half. Overlapping
  5 lines ensures context is preserved across boundaries.

WHAT METADATA DO WE ATTACH?
  Each chunk carries:
    - start_line / end_line  → which lines it covers (shown in UI)
    - error_count            → how many ERROR/CRITICAL lines are in it
    - warning_count          → how many WARNING lines are in it
    - chunk_index            → position in the document

  This metadata is displayed in the Streamlit UI so users understand
  exactly WHERE the retrieved context came from.
"""

import re

from langchain_core.documents import Document
from config.settings import CHUNK_LINES, OVERLAP_LINES

_WARN_RE = re.compile(r"\bWARN\b")   # matches WARN but NOT WARNING


def chunk_log_text(
    raw_text: str,
    chunk_lines: int = CHUNK_LINES,
    overlap_lines: int = OVERLAP_LINES,
) -> list[Document]:
    """
    Split raw log text into overlapping chunks of log lines.

    Args:
        raw_text:      The full content of the log file as a string.
        chunk_lines:   Number of lines per chunk (default: 20).
        overlap_lines: Lines shared between consecutive chunks (default: 5).

    Returns:
        List of LangChain Document objects, each with:
          - page_content : the chunk text (N log lines joined by newline)
          - metadata     : { start_line, end_line, error_count,
                             warning_count, chunk_index }

    Example:
        chunk_lines=20, overlap_lines=5 → step=15
        Chunk 0: lines  1-20
        Chunk 1: lines 16-35   (5-line overlap with chunk 0)
        Chunk 2: lines 31-50   (5-line overlap with chunk 1)
    """
    # Remove blank lines — they add nothing to the embedding
    all_lines = [line for line in raw_text.splitlines() if line.strip()]

    if not all_lines:
        return []

    step = max(1, chunk_lines - overlap_lines)  # how many lines to advance per chunk
    chunks: list[Document] = []

    for i in range(0, len(all_lines), step):
        window = all_lines[i : i + chunk_lines]
        content = "\n".join(window)

        # Count severity keywords for metadata (case-insensitive)
        upper_content = content.upper()
        error_count = upper_content.count("ERROR") + upper_content.count("CRITICAL") + upper_content.count("FATAL")
        warning_count = upper_content.count("WARNING") + len(_WARN_RE.findall(upper_content))

        chunks.append(
            Document(
                page_content=content,
                metadata={
                    "start_line": i + 1,
                    "end_line": i + len(window),
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "chunk_index": len(chunks),
                },
            )
        )

    return chunks
