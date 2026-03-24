"""
tests/test_pipeline.py
----------------------
Comprehensive test suite covering 10 scenarios.
Tests all pure-Python pipeline stages (no API key needed).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import tempfile
from pathlib import Path
from langchain_core.documents import Document

from analyzer.log_reader import (
    read_log_file, read_log_text, get_statistics, _parse_line,
)
from rag.chunker import chunk_log_text


SAMPLE_LOG = """2024-01-15 08:00:01 INFO  Application started. Version=2.4.1
2024-01-15 08:00:02 INFO  Connected to database: postgres://prod-db-01:5432/myapp
2024-01-15 08:05:03 WARNING  Disk usage at 78%
2024-01-15 08:08:00 ERROR  Database connection timeout
2024-01-15 08:08:05 ERROR  Database connection timeout (attempt 2)
2024-01-15 08:08:10 CRITICAL  Maximum retries exceeded. DB unavailable
2024-01-15 08:08:11 ERROR  Request failed: POST /api/checkout
2024-01-15 08:09:00 WARNING  Application in degraded mode
2024-01-15 08:11:30 INFO  Primary database recovered
2024-01-15 08:14:00 WARNING  Slow query detected (4.2s)
"""


class TestLogReader(unittest.TestCase):
    # ── Scenario 1: Parse standard log format ──────────────────────
    def test_01_parse_standard_log_format(self):
        """Standard ISO timestamp + level + message lines parse correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_LOG)
            tmp = f.name
        try:
            entries, summary = read_log_file(tmp)
            self.assertEqual(len(entries), 10)
            self.assertEqual(entries[0]["level"], "INFO")
            self.assertEqual(entries[3]["level"], "ERROR")
            self.assertEqual(entries[5]["level"], "CRITICAL")
            self.assertIn("Application started", entries[0]["message"])
            self.assertIn("=== Log File:", summary)
            print(f"  ✅ PASS  Scenario 1 — standard parse: {len(entries)} entries")
        finally:
            os.unlink(tmp)

    # ── Scenario 2: Non-standard lines fall back to UNKNOWN ────────
    def test_02_nonstandard_lines_fallback(self):
        """Lines without a recognised level are stored as UNKNOWN."""
        raw = "java.lang.NullPointerException\n\tat com.example.Foo.bar(Foo.java:42)\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            entries, _ = read_log_file(tmp)
            levels = [e["level"] for e in entries]
            # Both lines have no standard level — should be UNKNOWN
            self.assertTrue(all(l == "UNKNOWN" for l in levels))
            print(f"  ✅ PASS  Scenario 2 — non-standard lines → UNKNOWN ({levels})")
        finally:
            os.unlink(tmp)

    # ── Scenario 3: Empty file ──────────────────────────────────────
    def test_03_empty_file(self):
        """Empty log file returns zero entries, no crash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n\n\n")   # only blank lines
            tmp = f.name
        try:
            entries, summary = read_log_file(tmp)
            self.assertEqual(entries, [])
            print("  ✅ PASS  Scenario 3 — empty file → 0 entries, no crash")
        finally:
            os.unlink(tmp)

    # ── Scenario 4: Statistics merging ─────────────────────────────
    def test_04_statistics_merge(self):
        """WARN→WARNING and FATAL→CRITICAL are merged in get_statistics."""
        raw = (
            "2024-01-01 10:00:00 WARN  low disk\n"
            "2024-01-01 10:01:00 FATAL  process killed\n"
            "2024-01-01 10:02:00 WARNING  high mem\n"
            "2024-01-01 10:03:00 CRITICAL  db down\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            entries, _ = read_log_file(tmp)
            stats = get_statistics(entries)
            self.assertEqual(stats["WARNING"], 2)   # WARN + WARNING
            self.assertEqual(stats["CRITICAL"], 2)  # FATAL + CRITICAL
            self.assertNotIn("WARN", stats)
            self.assertNotIn("FATAL", stats)
            print(f"  ✅ PASS  Scenario 4 — stats merge: {stats}")
        finally:
            os.unlink(tmp)

    # ── Scenario 5: read_log_text line count ───────────────────────
    def test_05_read_log_text_line_count(self):
        """read_log_text returns correct non-blank line count."""
        raw = SAMPLE_LOG  # 10 non-blank lines
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            text, count = read_log_text(tmp)
            self.assertEqual(count, 10)
            self.assertIsInstance(text, str)
            self.assertIn("ERROR", text)
            print(f"  ✅ PASS  Scenario 5 — read_log_text: {count} lines, {len(text)} chars")
        finally:
            os.unlink(tmp)


class TestChunker(unittest.TestCase):
    # ── Scenario 6: Correct chunk count ────────────────────────────
    def test_06_chunk_count(self):
        """20-line log with chunk_lines=10, overlap=2 → step=8 → expected chunks."""
        lines = [f"2024-01-01 INFO Line {i}" for i in range(1, 21)]
        raw = "\n".join(lines)
        chunks = chunk_log_text(raw, chunk_lines=10, overlap_lines=2)
        # step=8: starts at 0,8,16 → 3 chunks (last partial chunk included)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIsInstance(chunks[0], Document)
        print(f"  ✅ PASS  Scenario 6 — chunk count: {len(chunks)} chunks for 20 lines (chunk=10,overlap=2)")

    # ── Scenario 7: Overlap is correct ─────────────────────────────
    def test_07_overlap_correctness(self):
        """Consecutive chunks share overlap_lines lines."""
        lines = [f"LINE_{i:03d}" for i in range(1, 41)]
        raw = "\n".join(lines)
        chunks = chunk_log_text(raw, chunk_lines=10, overlap_lines=3)
        # step = 10 - 3 = 7
        # Chunk 0: lines 1-10, Chunk 1: lines 8-17 → 3 shared lines (8,9,10)
        c0_lines = set(chunks[0].page_content.splitlines())
        c1_lines = set(chunks[1].page_content.splitlines())
        overlap = c0_lines & c1_lines
        self.assertEqual(len(overlap), 3,
            f"Expected 3 overlapping lines, got {len(overlap)}: {overlap}")
        print(f"  ✅ PASS  Scenario 7 — overlap: {len(overlap)} shared lines between chunk 0 and 1")

    # ── Scenario 8: Metadata accuracy ──────────────────────────────
    def test_08_metadata_accuracy(self):
        """error_count and warning_count in metadata are correct."""
        raw = (
            "INFO  all good\n"
            "ERROR  something broke\n"
            "CRITICAL  system down\n"
            "WARNING  disk low\n"
            "WARN  cpu high\n"
        )
        chunks = chunk_log_text(raw, chunk_lines=10, overlap_lines=0)
        self.assertEqual(len(chunks), 1)
        meta = chunks[0].metadata
        self.assertEqual(meta["error_count"], 2)    # ERROR + CRITICAL
        self.assertEqual(meta["warning_count"], 2)  # WARNING + WARN
        self.assertEqual(meta["start_line"], 1)
        self.assertEqual(meta["end_line"], 5)
        print(f"  ✅ PASS  Scenario 8 — metadata: errors={meta['error_count']}, "
              f"warnings={meta['warning_count']}, lines={meta['start_line']}-{meta['end_line']}")

    # ── Scenario 9: Empty input ─────────────────────────────────────
    def test_09_empty_input(self):
        """Empty or all-blank input returns empty list without crashing."""
        self.assertEqual(chunk_log_text(""), [])
        self.assertEqual(chunk_log_text("\n\n\n"), [])
        print("  ✅ PASS  Scenario 9 — empty input → [] with no crash")

    # ── Scenario 10: File smaller than chunk_lines ─────────────────
    def test_10_file_smaller_than_chunk_size(self):
        """A 3-line file with chunk_lines=20 produces exactly 1 chunk."""
        raw = "INFO  start\nERROR  fail\nINFO  end\n"
        chunks = chunk_log_text(raw, chunk_lines=20, overlap_lines=5)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["start_line"], 1)
        self.assertEqual(chunks[0].metadata["end_line"], 3)
        self.assertEqual(chunks[0].metadata["error_count"], 1)
        print(f"  ✅ PASS  Scenario 10 — 3-line file → 1 chunk "
              f"(lines {chunks[0].metadata['start_line']}-{chunks[0].metadata['end_line']})")


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  AI LOG ANALYZER — Test Suite (10 scenarios)")
    print("═" * 60 + "\n")
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None  # keep declaration order
    suite = loader.loadTestsFromTestCase(TestLogReader)
    suite.addTests(loader.loadTestsFromTestCase(TestChunker))
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)
    print()
    if result.wasSuccessful():
        print(f"  All {result.testsRun} tests passed ✅")
    else:
        print(f"  {len(result.failures)} failure(s), {len(result.errors)} error(s)")
        for test, tb in result.failures + result.errors:
            print(f"\n  ❌ FAIL: {test}")
            print(f"  {tb}")
    print("\n" + "═" * 60)
