from functools import lru_cache

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
    LLM_PROVIDER,
    OLLAMA_MODEL,
)


@lru_cache(maxsize=1)
def get_llm():
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not GEMINI_API_KEY:
            raise ValueError(
                "Set GEMINI_API_KEY in .env.local."
            )

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            thinking_level=GEMINI_THINKING_LEVEL,
        )

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
        )

    raise ValueError(
        "LLM_PROVIDER must be 'gemini' or 'ollama'."
    )