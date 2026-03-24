"""
app.py  —  Streamlit Web UI for the AI Log Analyzer
----------------------------------------------------
Run with:  streamlit run app.py

UI layout:
  ┌──────────────────┬───────────────────────────────────────┐
  │   SIDEBAR        │   MAIN AREA                           │
  │                  │                                       │
  │  📁 Upload file  │  📊 Stats row (lines/chunks/errors)   │
  │  [Use Sample]    │                                       │
  │  ──────────────  │  💡 Suggested questions (click to ask)│
  │  ⚙️ Settings     │                                       │
  │   Model select   │  💬 Chat history                      │
  │   Top-K slider   │    User question                      │
  │   Chunk size     │    AI answer (streamed live)           │
  │                  │    📎 Source chunks (expandable)       │
  │  [🔄 Index Log]  │                                       │
  │                  │  [Ask about your logs...]  ← input    │
  └──────────────────┴───────────────────────────────────────┘

SESSION STATE keys:
  vectorstore   — FAISS index (None until "Index Log" is clicked)
  chat_history  — list of { question, answer, sources }
  log_info      — { filename, total_lines, chunk_count, stats }
  raw_text      — current log file content as string
  filename      — display name of the loaded file
"""

import re
import streamlit as st
from pathlib import Path

from analyzer.log_reader import get_statistics
from rag.chunker import chunk_log_text
from rag.embedder import build_vectorstore
from rag.retriever import retrieve_with_scores
from rag.generator import stream_answer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Log Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_LOG_PATH = "sample_logs/app.log"

SUGGESTED_QUESTIONS = [
    "What critical errors occurred and what caused them?",
    "What was the timeline of the main incident?",
    "What are the top 3 recommendations to fix these issues?",
    "Are there any recurring error patterns?",
    "What warnings need immediate attention?",
    "Summarize the overall system health",
]

LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})?"
    r"\s*\[?(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\]?"
    r"\s*:?\s*(?P<message>.+)",
    re.IGNORECASE,
)

# ── Session State ─────────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "vectorstore":  None,
        "chat_history": [],
        "log_info":     None,
        "raw_text":     None,
        "filename":     None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ── Helper functions ──────────────────────────────────────────────────────────

def _parse_raw_to_entries(raw_text: str) -> list[dict]:
    """Parse raw log text into structured entries (no file I/O needed)."""
    entries = []
    for i, line in enumerate(raw_text.splitlines(), 1):
        if not line.strip():
            continue
        m = LOG_PATTERN.search(line)
        if m:
            entries.append({
                "line_no":   i,
                "timestamp": (m.group("timestamp") or "").strip(),
                "level":     (m.group("level") or "INFO").upper(),
                "message":   m.group("message").strip(),
                "raw":       line,
            })
        else:
            entries.append({
                "line_no": i, "timestamp": "", "level": "UNKNOWN",
                "message": line, "raw": line,
            })
    return entries


def _show_sources(docs_with_scores: list) -> None:
    """Render retrieved FAISS chunks in a collapsible expander."""
    if not docs_with_scores:
        return
    with st.expander(f"📎 {len(docs_with_scores)} source chunks retrieved from FAISS"):
        for i, item in enumerate(docs_with_scores, 1):
            doc, score = item if isinstance(item, tuple) else (item, None)
            start = doc.metadata.get("start_line", "?")
            end   = doc.metadata.get("end_line",   "?")
            errs  = doc.metadata.get("error_count",   0)
            warns = doc.metadata.get("warning_count", 0)
            score_txt = f" · distance {score:.3f}" if score is not None else ""
            st.markdown(
                f"**Chunk {i}** — Lines {start}–{end}{score_txt}"
                f" · 🔴 {errs} errors · 🟡 {warns} warnings"
            )
            st.code(doc.page_content, language="text")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 AI Log Analyzer")
    st.caption("LangChain · FAISS · Groq · Streamlit")
    st.divider()

    # ── File loading ──────────────────────────────────────────────
    st.subheader("📁 Load Log File")
    uploaded_file = st.file_uploader(
        "Upload a .log or .txt file",
        type=["log", "txt"],
        help="Any standard log file format",
    )
    if uploaded_file is not None:
        # Must read immediately — UploadedFile buffer is consumed on first read
        st.session_state.raw_text = uploaded_file.read().decode("utf-8", errors="replace")
        st.session_state.filename = uploaded_file.name

    col_load, col_clear = st.columns(2)
    if col_load.button("📄 Sample Log", use_container_width=True):
        if Path(SAMPLE_LOG_PATH).exists():
            st.session_state.raw_text = Path(SAMPLE_LOG_PATH).read_text(encoding="utf-8")
            st.session_state.filename = "app.log (sample)"
            st.success("Sample log loaded!")
        else:
            st.error(f"Not found: {SAMPLE_LOG_PATH}")

    if col_clear.button("🗑️ Clear", use_container_width=True):
        st.session_state.update({
            "vectorstore": None, "chat_history": [],
            "log_info": None, "raw_text": None, "filename": None,
        })
        st.rerun()

    if st.session_state.filename:
        st.info(f"Loaded: **{st.session_state.filename}**")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    selected_model = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        help="llama-3.3-70b: best quality · llama-3.1-8b: fastest · mixtral: balanced",
    )
    top_k = st.slider(
        "Chunks to retrieve (Top-K)", min_value=1, max_value=10, value=5,
        help="Number of log chunks fed to the LLM per question",
    )
    chunk_lines = st.slider(
        "Lines per chunk", min_value=5, max_value=50, value=20,
        help="Larger = more context per chunk, fewer total chunks",
    )
    st.divider()

    # ── Index button ──────────────────────────────────────────────
    if st.button(
        "🔄  Index Log",
        type="primary",
        use_container_width=True,
        disabled=(st.session_state.raw_text is None),
    ):
        raw = st.session_state.raw_text

        with st.spinner("Step 1/3 — Chunking log lines..."):
            chunks = chunk_log_text(raw, chunk_lines=chunk_lines)

        with st.spinner(f"Step 2/3 — Embedding {len(chunks)} chunks (local model)..."):
            try:
                vs = build_vectorstore(chunks)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        with st.spinner("Step 3/3 — Finalizing FAISS index..."):
            entries = _parse_raw_to_entries(raw)
            stats = get_statistics(entries)
            line_count = len([l for l in raw.splitlines() if l.strip()])

        st.session_state.vectorstore  = vs
        st.session_state.chat_history = []        # clear history on re-index
        st.session_state.log_info = {
            "filename":    st.session_state.filename,
            "total_lines": line_count,
            "chunk_count": len(chunks),
            "stats":       stats,
        }
        st.success(f"✅ Indexed {len(chunks)} chunks from {line_count} lines!")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.title("🔍 AI Log Analyzer")
st.caption(
    "RAG pipeline: **Logs → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer**"
)

# ── Empty state ───────────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.info("👈  Upload a log file (or click **Sample Log**) then click **Index Log** to begin.")

    with st.expander("ℹ️  How the RAG pipeline works"):
        st.markdown("""
```
1. CHUNKING     log_reader → chunk_log_text()
                Split the log file into 20-line windows with 5-line overlap.
                Each chunk = one searchable unit in the vector DB.

2. EMBEDDING    embedder.build_vectorstore()
                Send each chunk to all-MiniLM-L6-v2 (local model).
                Returns a 384-dim float vector per chunk.
                Store vectors + chunk text in FAISS (in-memory).

3. RETRIEVAL    retriever.retrieve_with_scores()   (at query time)
                Embed the user's question → compare against all chunk vectors
                → return the TOP-K most similar chunks (cosine similarity).

4. GENERATION   generator.stream_answer()
                Build prompt:  system_prompt + retrieved_chunks + question
                Stream GPT-4o-mini response token by token.
```
**Why RAG instead of sending the whole log?**
Log files can be thousands of lines. RAG retrieves only the *relevant* chunks,
keeping prompts short, answers focused, and costs low.
        """)

# ── Indexed state ─────────────────────────────────────────────────────────────
else:
    info  = st.session_state.log_info
    stats = info["stats"]

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📄 Log Lines",  info["total_lines"])
    c2.metric("🧩 Chunks",     info["chunk_count"])
    c3.metric("🔴 Errors",     stats.get("ERROR", 0) + stats.get("CRITICAL", 0))
    c4.metric("🟡 Warnings",   stats.get("WARNING", 0))
    c5.metric("🟢 Info",       stats.get("INFO", 0))

    st.divider()

    # Suggested questions
    st.subheader("💡 Quick Questions  —  click to ask instantly")
    pending_question = None
    for row_questions in [SUGGESTED_QUESTIONS[:3], SUGGESTED_QUESTIONS[3:]]:
        cols = st.columns(3)
        for col, q in zip(cols, row_questions):
            if col.button(q, use_container_width=True, key=f"sq_{q[:25]}"):
                pending_question = q

    st.divider()

    # Render chat history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            _show_sources(item["sources"])

    # Accept query from chat input OR suggested question button
    typed_query = st.chat_input("Ask anything about your logs...")
    query = typed_query or pending_question

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS for relevant chunks..."):
                docs_with_scores = retrieve_with_scores(
                    st.session_state.vectorstore, query, k=top_k
                )

            # Stream the LLM answer token by token
            docs = [d for d, _ in docs_with_scores]
            answer = st.write_stream(stream_answer(query, docs, model=selected_model))

            # Show which chunks were used
            _show_sources(docs_with_scores)

        # Persist to chat history
        st.session_state.chat_history.append({
            "question": query,
            "answer":   answer,
            "sources":  docs_with_scores,
        })
