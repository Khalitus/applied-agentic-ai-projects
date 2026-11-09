"""Raw-data integrity checks. Keep source columns intact for auditing."""
import pandas as pd

from src.config import RAW_CSV, RAW_FEATURES, TARGET
EXPECTED = {"instant", "dteday", "casual", "registered", TARGET, *RAW_FEATURES}

def load_hourly(path=RAW_CSV) -> pd.DataFrame:
    if not path.exists(): 
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m scripts.download_data"
        )
    df = pd.read_csv(path)
    missing = EXPECTED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing UCI source columns: {sorted(missing)}")
    if df["instant"].duplicated().any():
        raise ValueError("Duplicate source record IDs detected")
    if df[list(RAW_FEATURES) + [TARGET]].isna().any().any():
        raise ValueError("Missing values in model inputs/target")
    if not (df["casual"] + df["registered"] == df[TARGET]).all():
        raise ValueError("Target count does not equal casual + registered")
    return df.sort_values(["dteday", "hr"], kind="stable").reset_index(drop=True)
