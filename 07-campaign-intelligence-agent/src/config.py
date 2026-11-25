from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "bank-full.csv"
DATABASE = ROOT / "data" / "campaigns.sqlite3"
MODEL = ROOT / "artifacts" / "subscription_model.joblib"
METRICS = ROOT / "reports" / "baseline_metrics.json"
