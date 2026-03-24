"""
config/settings.py
------------------
Single source of truth for all configuration.
Change model, chunk size, or retrieval settings here — nowhere else.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Use absolute path so .env is found regardless of working directory
load_dotenv(Path(__file__).parent.parent / ".env")

# ── API ────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ── Models ─────────────────────────────────────────────────────────
# LLM used for generating the final answer (Groq — free tier)
MODEL     = "llama-3.3-70b-versatile"   # swap to "llama-3.1-8b-instant" for faster/cheaper
LLM_MODEL = MODEL                        # alias used by rag/generator.py
MAX_TOKENS = 2048

# Embedding model — runs locally via sentence-transformers (no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dims, fast, free, offline

# ── RAG Pipeline ───────────────────────────────────────────────────
# How many log LINES go into each chunk sent to the vector store
CHUNK_LINES = 20

# How many lines of overlap between consecutive chunks
# Overlap prevents losing context at chunk boundaries
OVERLAP_LINES = 5

# How many chunks to retrieve from FAISS for each question
TOP_K = 5

# ── Display ────────────────────────────────────────────────────────
REPORT_WIDTH = 70
