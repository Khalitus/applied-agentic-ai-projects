from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "hour.csv"
DB_PATH = DATA_DIR / "processed" / "bike_demand.sqlite3"
MODEL_PATH = ROOT / "models" / "demand_xgb.joblib"
REPORT_PATH = ROOT / "reports" / "model_metrics.json"

TARGET = "cnt"
FEATURES = (
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
)
