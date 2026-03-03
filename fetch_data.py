#!/usr/bin/env python3
"""
Fetch multi-decade daily weather data for Melbourne from Open-Meteo Historical API.

Run this locally (it needs internet access to Open-Meteo):
    python fetch_data.py

The script will download 1950-2025 daily data in 5-year chunks and save to:
    data/melbourne_daily.parquet
    data/melbourne_daily.csv

Open-Meteo variables fetched:
  - temperature_2m_max, temperature_2m_min, temperature_2m_mean
  - apparent_temperature_max, apparent_temperature_min, apparent_temperature_mean
  - precipitation_sum, rain_sum, precipitation_hours
  - windspeed_10m_max, windgusts_10m_max, winddirection_10m_dominant
  - shortwave_radiation_sum  (total daily solar energy, MJ/m²)
  - daylight_duration        (seconds of daylight)
  - sunshine_duration        (seconds of actual sunshine)
  - et0_fao_evapotranspiration

Melbourne Olympic Park station area: -37.8136, 144.9631
No API key required.
"""

import requests
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

MELBOURNE_LAT = -37.8136
MELBOURNE_LON = 144.9631

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
    "shortwave_radiation_sum",
    "daylight_duration",
    "sunshine_duration",
    "et0_fao_evapotranspiration",
]


def fetch_chunk(start_date: str, end_date: str) -> dict:
    """Fetch one date-range chunk from Open-Meteo with retries."""
    params = {
        "latitude": MELBOURNE_LAT,
        "longitude": MELBOURNE_LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Australia/Melbourne",
    }
    for attempt in range(5):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            print(f"  HTTP {resp.status_code} on attempt {attempt+1}, retrying...")
        except requests.RequestException as e:
            print(f"  Request error on attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)
    # Final attempt - raise on failure
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_melbourne_data(start_year: int = 1950, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch daily weather for Melbourne across many decades.
    Chunks into 5-year blocks to stay within API limits.
    """
    all_frames = []
    chunk_size = 5  # years per request

    for y in range(start_year, end_year + 1, chunk_size):
        y_end = min(y + chunk_size - 1, end_year)
        sd = f"{y}-01-01"
        ed = f"{y_end}-12-31"
        print(f"Fetching {sd} to {ed} ...")
        data = fetch_chunk(sd, ed)
        daily = data["daily"]
        df_chunk = pd.DataFrame(daily)
        all_frames.append(df_chunk)
        time.sleep(1)  # polite rate-limiting

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived weather variables useful for clustering."""
    df["daylight_hours"] = df["daylight_duration"] / 3600.0
    df["sunshine_hours"] = df["sunshine_duration"] / 3600.0
    df["sunshine_fraction"] = (
        df["sunshine_hours"] / df["daylight_hours"].replace(0, float("nan"))
    )
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "melbourne_daily.parquet"
    csv_path = out_dir / "melbourne_daily.csv"

    start_year = 1950
    end_year = 2025

    print(f"=== Fetching Melbourne weather data ({start_year}-{end_year}) ===")
    print(f"Source: Open-Meteo Historical Weather API")
    print(f"Location: Melbourne ({MELBOURNE_LAT}, {MELBOURNE_LON})")
    print(f"Variables: {len(DAILY_VARIABLES)} raw + derived\n")

    df = fetch_melbourne_data(start_year, end_year)
    df = add_derived_columns(df)

    print(f"\nFetched {len(df)} days, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSummary stats:")
    print(df.describe().round(2))

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {parquet_path} and {csv_path}")
    print("Done! Now run: python find_seasons.py")


if __name__ == "__main__":
    main()
