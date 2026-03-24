"""
analyzer/log_analyzer.py
------------------------
STEP 2: Send the parsed logs to Grok (xAI) and get an AI-powered analysis.

WHAT IT DOES:
  - Builds a structured prompt with clear instructions for Grok
  - Uses streaming so the user sees output immediately (no long wait)
  - Returns the full analysis text once streaming completes

WHY USE STREAMING?
  We use streaming here for two reasons:
   1. UX — the user sees output in real time instead of waiting 10+ seconds
   2. Reliability — streaming avoids HTTP timeout issues on large outputs.

WHY OPENAI SDK FOR GROK?
  xAI's Grok API is OpenAI-compatible — same request format, same SDK.
  We just swap the base_url to point at xAI instead of OpenAI.
"""

from openai import OpenAI
from config.settings import GROQ_API_KEY, GROQ_BASE_URL, MODEL, MAX_TOKENS

# System prompt — tells Grok exactly WHAT role to play and HOW to structure output.
# This is the most important part of the prompt engineering.
SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) and log analysis specialist.

Your job is to analyze application log files and provide actionable insights.

Always structure your response EXACTLY as follows:

## 1. Executive Summary
A 2-3 sentence overview of the overall health of the system based on the logs.

## 2. Critical Issues
List any ERROR, CRITICAL, or FATAL log entries. For each:
- Quote the relevant log line
- Explain what it means in plain English
- Suggest a likely root cause
If none, write "No critical issues found."

## 3. Warnings & Anomalies
List WARNING-level entries and any suspicious patterns.
If none, write "No warnings found."

## 4. Patterns & Trends
Identify any repeating errors, time-based patterns, or cascading failures.
Example: "Database connection errors spike every 30 minutes"

## 5. Actionable Recommendations
Numbered list of concrete next steps the engineering team should take.
Be specific — mention file names, services, or line numbers from the logs.

## 6. Severity Score
Rate the overall log severity: LOW / MEDIUM / HIGH / CRITICAL
Explain your rating in one sentence.
"""


def analyze_logs(log_summary: str, stream_to_console: bool = True) -> str:
    """
    Send the log summary to Grok and return the AI analysis.

    Args:
        log_summary:        The formatted log text from log_reader.py
        stream_to_console:  If True, print tokens in real-time as they arrive

    Returns:
        The full analysis text as a string.

    Raises:
        ValueError: if XAI_API_KEY is not set
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "1. Get a free key at https://console.groq.com/\n"
            "2. Add it to your .env file: GROQ_API_KEY=your-key-here"
        )

    # Initialize the OpenAI client pointed at Groq's API endpoint.
    # Groq uses the same API format as OpenAI — only the base_url differs.
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )

    # Build the user message — this contains the actual log data.
    user_message = f"""Please analyze the following log file and provide a detailed report:

{log_summary}

Focus on errors that could impact users or system stability.
Reference specific line numbers when discussing issues.
"""

    # Use streaming so output appears immediately.
    full_text = ""
    with client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    ) as stream:
        for chunk in stream:
            text_chunk = chunk.choices[0].delta.content or ""
            if stream_to_console:
                print(text_chunk, end="", flush=True)
            full_text += text_chunk

    return full_text
