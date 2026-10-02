from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.local")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
EVAL_DIR = DATA_DIR / "evaluation"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DB_PATH = ARTIFACTS_DIR / "support.db"
MODEL_PATH = ARTIFACTS_DIR / "escalation_pipeline.joblib"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"

COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini",
).lower()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

GEMINI_THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "low",
).lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)