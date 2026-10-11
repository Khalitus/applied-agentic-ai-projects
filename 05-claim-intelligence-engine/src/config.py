import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env.local")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

CLAIMS_PATH = RAW_DATA_DIR / "claims.csv"
DATABASE_PATH = DATABASE_DIR / "claims.db"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "claim_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

KNOWLEDGE_DIR = ROOT_DIR / "knowledge"

POLICY_DIR = ROOT_DIR / "data" / "policies"
VECTOR_STORE_DIR = ROOT_DIR / "data" / "vector_store"

TARGET_COLUMN = "claim_amount"
RANDOM_STATE = 42
TEST_SIZE = 0.20

TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "raw").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

MULTIMODAL_PROVIDER = os.getenv("MULTIMODAL_PROVIDER", "local").strip().lower()
LOCAL_IMAGE_MODEL = os.getenv(
    "LOCAL_IMAGE_MODEL",
    "Salesforce/blip-image-captioning-base",
).strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
).strip()

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))