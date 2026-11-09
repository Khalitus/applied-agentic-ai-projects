"""Check primary and fallback provider access without disclosing credentials."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from groq import Groq


def _show_failure(provider: str, error: Exception) -> None:
    status = getattr(error, "code", None)
    if not isinstance(status, int):
        status = getattr(error, "status_code", None)
    print(f"{provider}: {type(error).__name__} (HTTP {status or 'unknown'})")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not gemini_key:
        print("Gemini: GEMINI_API_KEY not configured")
    else:
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        try:
            reply = genai.Client(api_key=gemini_key).models.generate_content(
                model=model, contents="Reply with exactly the word READY."
            )
            print(f"Gemini: available ({model}); received response={bool(reply.text)}")
        except Exception as error:
            _show_failure("Gemini", error)

    if not groq_key:
        print("Groq: GROQ_API_KEY not configured")
    else:
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        try:
            reply = Groq(api_key=groq_key, timeout=30.0, max_retries=0).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly the word READY."}],
                max_completion_tokens=100,
            )
            text = reply.choices[0].message.content
            print(f"Groq: available ({model}); received response={bool(text)}")
        except Exception as error:
            _show_failure("Groq", error)


if __name__ == "__main__":
    main()
