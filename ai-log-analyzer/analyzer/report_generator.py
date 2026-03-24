"""
analyzer/report_generator.py
-----------------------------
STEP 3: Format and display the analysis results in a readable terminal report.

WHAT IT DOES:
  - Prints a header with file stats (line counts per level)
  - Streams the AI analysis to the terminal in real time
  - Prints a footer with token usage and cost estimate

ADDITIONAL CONSIDERATIONS:
  If we wanted to add JSON output, save to file, or send via Slack,
  we'd add that here without touching the reader or analyzer.
  Clean separation of concerns.
"""

from config.settings import REPORT_WIDTH


def print_header(file_path: str, stats: dict, total_lines: int) -> None:
    """
    Print the report banner with file info and log level breakdown.

    Example output:
    ══════════════════════════════════════════════════════════════════
    AI LOG ANALYZER  |  Powered by Claude claude-opus-4-6
    ══════════════════════════════════════════════════════════════════
    File     : sample_logs/app.log
    Lines    : 47 analyzed
    Breakdown: ERROR=5  WARNING=3  INFO=35  DEBUG=4
    ──────────────────────────────────────────────────────────────────
    """
    separator_heavy = "═" * REPORT_WIDTH
    separator_light = "─" * REPORT_WIDTH

    print(f"\n{separator_heavy}")
    print(f"  AI LOG ANALYZER  |  Powered by Claude claude-opus-4-6")
    print(separator_heavy)
    print(f"  File     : {file_path}")
    print(f"  Lines    : {total_lines} analyzed")

    # Build a compact level breakdown string: ERROR=5  WARNING=3  INFO=35
    breakdown_parts = [f"{level}={count}" for level, count in stats.items()]
    print(f"  Breakdown: {'  '.join(breakdown_parts)}")
    print(separator_light)
    print()


def print_analysis_header() -> None:
    """Print the section header before streaming starts."""
    print("  AI ANALYSIS (streaming...)")
    print("─" * REPORT_WIDTH)
    print()


def print_footer(usage) -> None:
    """
    Print token usage and estimated cost after the analysis.

    usage is an anthropic.types.Usage object with:
      input_tokens  - tokens sent to Claude (your prompt + logs)
      output_tokens - tokens Claude generated (the analysis)
    """
    separator_light = "─" * REPORT_WIDTH
    separator_heavy = "═" * REPORT_WIDTH

    # Rough cost estimate using claude-opus-4-6 pricing:
    # Input: $5.00 per 1M tokens  →  $0.000005 per token
    # Output: $25.00 per 1M tokens →  $0.000025 per token
    input_cost = usage.input_tokens * 0.000005
    output_cost = usage.output_tokens * 0.000025
    total_cost = input_cost + output_cost

    print()
    print(separator_light)
    print(f"  TOKEN USAGE")
    print(f"  Input tokens  : {usage.input_tokens:,}  (${input_cost:.4f})")
    print(f"  Output tokens : {usage.output_tokens:,}  (${output_cost:.4f})")
    print(f"  Total cost    : ~${total_cost:.4f} USD")
    print(separator_heavy)
    print()


def save_report_to_file(file_path: str, header_text: str, analysis_text: str) -> None:
    """
    Optionally save the full report to a .txt file.

    Args:
        file_path:     Where to save (e.g., "report.txt")
        header_text:   The pre-analysis summary (file stats)
        analysis_text: The AI analysis text
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(header_text)
        f.write("\n\nAI ANALYSIS:\n")
        f.write("─" * REPORT_WIDTH + "\n")
        f.write(analysis_text)
    print(f"\n  Report saved to: {file_path}")
