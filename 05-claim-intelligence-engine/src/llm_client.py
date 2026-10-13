import os

from dotenv import load_dotenv

from src.config import ROOT_DIR


load_dotenv(ROOT_DIR / ".env.local")

def generate_with_groq(
    system_prompt: str,
    user_prompt: str,
) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    client = Groq(
        api_key=api_key,
        timeout=60.0,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        ),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=1800,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    return content

def generate_with_gemini(
    system_prompt: str,
    user_prompt: str,
) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            ),
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=1800,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text
    finally:
        client.close()

def generate_llm_response(
    system_prompt: str,
    user_prompt: str,
) -> str:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        return generate_with_groq(system_prompt, user_prompt)

    if provider == "gemini":
        return generate_with_gemini(system_prompt, user_prompt)

    raise ValueError(f"Unsupported LLM provider: {provider}")