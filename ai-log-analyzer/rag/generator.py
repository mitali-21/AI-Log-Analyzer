"""
rag/generator.py
----------------
PIPELINE STEP 4 — Generate the final answer using the LLM + retrieved context.

THE RAG PROMPT PATTERN:
  We combine two things into the LLM prompt:
    1. The retrieved log chunks (the "context")
    2. The user's question

  The system prompt instructs the LLM to ONLY use the provided context
  and to cite line numbers. This prevents hallucination — the LLM can't
  invent log entries that don't exist.

  Prompt structure:
    ┌─────────────────────────────────────────────┐
    │ SYSTEM: You are an SRE expert. Answer ONLY  │
    │         from the log context provided.      │
    │                                             │
    │ HUMAN:  Log Context:                        │
    │         [Chunk 1 — Lines 11-30]             │
    │         2024-01-15 08:08:10 CRITICAL ...    │
    │         ...                                 │
    │         [Chunk 2 — Lines 1-20]              │
    │         ...                                 │
    │                                             │
    │         Question: What caused the outage?   │
    └─────────────────────────────────────────────┘

TWO MODES:
  - generate_answer()  → non-streaming, returns full string (used by CLI)
  - stream_answer()    → generator, yields tokens (used by Streamlit)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config.settings import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL

# ── Prompt Template ────────────────────────────────────────────────────────────
# {context}  is filled with the retrieved log chunks
# {question} is filled with the user's query
SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) and log analysis specialist.

Your task is to answer questions about a log file using ONLY the log context provided below.
Do NOT invent or assume log entries that are not in the context.

When answering:
- Cite specific line numbers (e.g. "Line 14 shows...")
- Be direct and actionable
- If the context doesn't contain enough information to answer, say so clearly
- Highlight severity (CRITICAL > ERROR > WARNING > INFO)
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Log Context:\n{context}\n\nQuestion: {question}"),
])


def _build_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a readable context block.
    Each chunk is labeled with its line range so the LLM can reference them.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        start = doc.metadata.get("start_line", "?")
        end   = doc.metadata.get("end_line", "?")
        parts.append(f"[Chunk {i} — Lines {start}–{end}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _make_llm(model: str, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,          # deterministic — we want factual answers, not creative ones
        streaming=streaming,
        openai_api_key=GROQ_API_KEY,
        openai_api_base=GROQ_BASE_URL,
    )


# ── Non-streaming (CLI) ────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    docs: list[Document],
    model: str = LLM_MODEL,
) -> str:
    """
    Generate a complete answer in one shot (non-streaming).
    Used by the CLI (main.py).

    Returns the answer as a plain string.
    """
    context = _build_context(docs)
    chain   = PROMPT_TEMPLATE | _make_llm(model, streaming=False)
    response = chain.invoke({"context": context, "question": query})
    return response.content


# ── Streaming (Streamlit) ──────────────────────────────────────────────────────

def stream_answer(
    query: str,
    docs: list[Document],
    model: str = LLM_MODEL,
):
    """
    Stream the answer token by token.
    Used by Streamlit's st.write_stream().

    Yields:
        str — one token (or small chunk of text) at a time

    Usage in Streamlit:
        answer = st.write_stream(stream_answer(query, docs))
        # `answer` holds the full text once streaming completes
    """
    context = _build_context(docs)
    chain   = PROMPT_TEMPLATE | _make_llm(model, streaming=True)

    for chunk in chain.stream({"context": context, "question": query}):
        yield chunk.content
