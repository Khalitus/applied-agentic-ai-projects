from config import LLM_PROVIDER, OLLAMA_MODEL, OPENAI_MODEL


def get_llm():
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        if not OPENAI_MODEL:
            raise ValueError("Set OPENAI_MODEL in .env.")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'.")
