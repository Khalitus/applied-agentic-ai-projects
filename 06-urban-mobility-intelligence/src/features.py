"""Shared feature contract for model training and user-submitted scenarios."""
from datetime import date

import pandas as pd

from src.config import FEATURES


def training_features(df: pd.DataFrame) -> pd.DataFrame:
    # Never include casual or registered: together they reveal the target exactly.
    return df.loc[:, list(FEATURES)].copy()


def scenario_features(
    date_text: str,
    hour: int,
    holiday: bool,
    weather_code: int,
    temperature_c: float,
    humidity_pct: float,
    windspeed_kmh: float,
) -> pd.DataFrame:
    day = date.fromisoformat(date_text)
    if not 0 <= int(hour) <= 23:
        raise ValueError("Hour must be between 0 and 23")
    if int(weather_code) not in (1, 2, 3, 4):
        raise ValueError("Weather category must be 1, 2, 3 or 4")
    if not 0 <= humidity_pct <= 100:
        raise ValueError("Humidity must be between 0 and 100 percent")
    if not 0 <= temperature_c <= 41:
        raise ValueError("Temperature must be between 0 and 41 °C for this demo")
    if not 0 <= windspeed_kmh <= 67:
        raise ValueError("Wind speed must be between 0 and 67 km/h")

    # UCI Sunday=0; Python date.weekday() uses Monday=0.
    weekday = (day.weekday() + 1) % 7
    workingday = int(weekday not in (0, 6) and not bool(holiday))
    values = {
        "mnth": day.month,
        "hr": int(hour),
        "holiday": int(holiday),
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": int(weather_code),
        # Approximate physical-unit conversions following dataset source notes.
        "temp": temperature_c / 41.0,
        "hum": humidity_pct / 100.0,
        "windspeed": windspeed_kmh / 67.0,
    }
    return pd.DataFrame([values], columns=list(FEATURES))
