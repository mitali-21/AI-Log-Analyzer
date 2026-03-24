"""
main.py  —  CLI entry point for the AI Log Analyzer (RAG pipeline)
-------------------------------------------------------------------
Usage:
    python3 main.py sample_logs/app.log "What caused the database outage?"
    python3 main.py /var/log/nginx/error.log "Show all critical errors"

For the full interactive web UI, run:
    streamlit run app.py
"""

import sys
import argparse

from analyzer.log_reader import read_log_text, read_log_file, get_statistics
from rag.chunker import chunk_log_text
from rag.embedder import build_vectorstore
from rag.retriever import retrieve_relevant_chunks
from rag.generator import generate_answer
from config.settings import REPORT_WIDTH


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-powered log analyzer (RAG: LangChain + FAISS + OpenAI)",
        epilog='Example: python3 main.py sample_logs/app.log "What caused the outage?"',
    )
    parser.add_argument("log_file",  help="Path to the log file")
    parser.add_argument("question",  help='Question to ask, e.g. "What are the critical errors?"')
    parser.add_argument("--top-k",   type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--model",   default="llama-3.3-70b-versatile", help="Groq model to use")
    parser.add_argument("--chunk-lines", type=int, default=20, help="Lines per chunk (default: 20)")
    return parser.parse_args()


def separator(heavy=False):
    ch = "═" if heavy else "─"
    print(ch * REPORT_WIDTH)


def main():
    args = parse_args()

    # ── Step 1: Read log file ──────────────────────────────────────
    print(f"\nReading: {args.log_file}")
    try:
        raw_text, line_count = read_log_text(args.log_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    entries, _ = read_log_file(args.log_file)
    stats = get_statistics(entries)

    separator(heavy=True)
    print(f"  AI LOG ANALYZER  (RAG)  |  {args.model}")
    separator(heavy=True)
    print(f"  File    : {args.log_file}")
    print(f"  Lines   : {line_count}")
    breakdown = "  ".join(f"{k}={v}" for k, v in stats.items())
    print(f"  Levels  : {breakdown}")
    separator()

    # ── Step 2: Chunk ──────────────────────────────────────────────
    print(f"\n  [1/4] Chunking into {args.chunk_lines}-line windows...")
    chunks = chunk_log_text(raw_text, chunk_lines=args.chunk_lines)
    print(f"        → {len(chunks)} chunks created")

    # ── Step 3: Embed + Index ──────────────────────────────────────
    print(f"  [2/4] Embedding chunks with text-embedding-3-small...")
    try:
        vectorstore = build_vectorstore(chunks)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nEmbedding Error: {e}")
        sys.exit(1)
    print(f"        → FAISS index ready")

    # ── Step 4: Retrieve ───────────────────────────────────────────
    print(f"  [3/4] Retrieving top-{args.top_k} chunks for your question...")
    docs = retrieve_relevant_chunks(vectorstore, args.question, k=args.top_k)
    print(f"        → {len(docs)} chunks retrieved")

    # ── Step 5: Generate ───────────────────────────────────────────
    print(f"  [4/4] Generating answer with {args.model}...\n")
    separator()
    print(f"  Question: {args.question}")
    separator()
    print()

    try:
        answer = generate_answer(args.question, docs, model=args.model)
    except Exception as e:
        print(f"\nGeneration Error: {e}")
        sys.exit(1)

    print(answer)

    # ── Sources ────────────────────────────────────────────────────
    separator()
    print(f"  Source chunks used ({len(docs)}):")
    for i, doc in enumerate(docs, 1):
        s = doc.metadata.get("start_line", "?")
        e = doc.metadata.get("end_line", "?")
        errs = doc.metadata.get("error_count", 0)
        print(f"  [{i}] Lines {s}–{e}  |  errors={errs}")
    separator(heavy=True)
    print()
    print("  Tip: run  streamlit run app.py  for the interactive web UI")
    print()


if __name__ == "__main__":
    main()
