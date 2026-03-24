"""
analyzer/log_reader.py
----------------------
STEP 1: Read a log file and parse it into structured data.

Responsibilities:
  - Read file from disk
  - Parse each line into { line_no, timestamp, level, message }
  - Expose raw text for the RAG chunker
  - Return statistics (how many ERRORs, WARNINGs, etc.)
"""

import re
from pathlib import Path

# Matches the most common log formats:
#   2024-01-15 10:23:45 ERROR  Something went wrong
#   [2024-01-15T10:23:45] WARN  Disk space low
#   INFO: Server started on port 8080
LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})?"
    r"\s*\[?(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\]?"
    r"\s*:?\s*(?P<message>.+)",
    re.IGNORECASE,
)


def read_log_file(file_path: str) -> tuple[list[dict], str]:
    """
    Read a log file and return:
      - entries : list of parsed dicts  { line_no, timestamp, level, message }
      - summary : formatted text block  (legacy: used by old Claude-based analyzer)

    Raises FileNotFoundError if the path doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries = []

    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        entries.append(_parse_line(line_no, stripped))

    summary_text = _build_summary_text(entries, path.name)
    return entries, summary_text


def read_log_text(file_path: str) -> tuple[str, int]:
    """
    NEW (for RAG pipeline): return the raw log text and line count.
    The chunker will split this text into overlapping chunks.

    Returns:
        (raw_text, line_count)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    line_count = len([l for l in raw_text.splitlines() if l.strip()])
    return raw_text, line_count


def _parse_line(line_no: int, raw_line: str) -> dict:
    match = LOG_PATTERN.search(raw_line)
    if match:
        return {
            "line_no": line_no,
            "timestamp": (match.group("timestamp") or "").strip(),
            "level": (match.group("level") or "INFO").upper(),
            "message": match.group("message").strip(),
            "raw": raw_line,
        }
    return {
        "line_no": line_no,
        "timestamp": "",
        "level": "UNKNOWN",
        "message": raw_line,
        "raw": raw_line,
    }


def _build_summary_text(entries: list[dict], filename: str) -> str:
    lines = [f"=== Log File: {filename} ==="]
    for entry in entries:
        prefix = f"[Line {entry['line_no']}]"
        ts = f" {entry['timestamp']}" if entry["timestamp"] else ""
        lines.append(f"{prefix}{ts} {entry['level']}: {entry['message']}")
    return "\n".join(lines)


def get_statistics(entries: list[dict]) -> dict:
    """Count log entries by level. Merges WARN→WARNING, FATAL→CRITICAL."""
    counts: dict[str, int] = {}
    for entry in entries:
        lvl = entry["level"]
        counts[lvl] = counts.get(lvl, 0) + 1

    counts["WARNING"] = counts.pop("WARNING", 0) + counts.pop("WARN", 0)
    counts["CRITICAL"] = counts.pop("CRITICAL", 0) + counts.pop("FATAL", 0)
    return {k: v for k, v in counts.items() if v > 0}
