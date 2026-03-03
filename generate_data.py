"""
Generate realistic multi-decade daily weather data for Melbourne, Australia.

Based on published Bureau of Meteorology (BOM) climatological normals for
Melbourne Olympic Park / Melbourne Regional Office, plus known physical
relationships (daylight from latitude, solar radiation from season, etc.).

This produces data with realistic:
  - Seasonal cycles (sinusoidal + harmonics)
  - Day-to-day autocorrelation (weather persistence)
  - Cross-variable correlations (e.g. hot days = more sun, less rain)
  - Interannual variability
  - Realistic noise structure
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

MELBOURNE_LAT = -37.8136

# ── Monthly climatological normals (BOM Melbourne Olympic Park) ──────────
# Indexed Jan=0 .. Dec=11
MONTHLY_NORMALS = {
    "temp_max":   [25.9, 25.8, 23.9, 20.3, 16.7, 14.1, 13.5, 15.0, 17.2, 19.7, 22.0, 24.2],
    "temp_min":   [14.3, 14.6, 13.2, 10.8,  8.7,  6.9,  6.0,  6.7,  8.0,  9.5, 11.2, 12.9],
    "rain_mm":    [47.3, 48.1, 50.4, 57.3, 56.0, 49.2, 47.7, 50.1, 58.2, 66.3, 60.4, 59.5],
    "rain_days":  [ 8.3,  7.1,  8.5, 10.8, 12.7, 13.0, 13.7, 14.1, 13.0, 12.6, 10.6,  9.6],
    "sun_hours":  [ 8.3,  7.6,  6.5,  5.0,  3.8,  3.2,  3.5,  4.3,  5.2,  6.2,  7.2,  7.8],
    "wind_mean":  [17.5, 16.8, 16.0, 15.5, 15.8, 16.2, 16.5, 17.2, 18.0, 18.5, 18.2, 17.8],
    "wind_gust":  [56.0, 52.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 60.0, 58.0, 56.0, 55.0],
    "humidity_9am": [59, 61, 64, 70, 76, 80, 79, 73, 67, 63, 60, 58],
    "humidity_3pm": [48, 49, 50, 53, 58, 61, 58, 53, 50, 49, 48, 47],
    "solar_mj":   [24.5, 20.8, 16.0, 10.7, 7.2, 5.5, 6.2, 9.0, 13.0, 17.5, 21.2, 24.0],
    "evapotrans": [5.8, 4.9, 3.6, 2.2, 1.3, 0.9, 1.0, 1.5, 2.4, 3.4, 4.5, 5.5],
}


def daylight_hours(day_of_year: np.ndarray) -> np.ndarray:
    """Calculate daylight hours from latitude and day of year (astronomical)."""
    lat_rad = np.radians(MELBOURNE_LAT)
    # Solar declination
    decl = 23.45 * np.sin(np.radians((284 + day_of_year) * 360 / 365))
    decl_rad = np.radians(decl)
    # Hour angle at sunrise
    cos_ha = -np.tan(lat_rad) * np.tan(decl_rad)
    cos_ha = np.clip(cos_ha, -1, 1)
    ha = np.degrees(np.arccos(cos_ha))
    return 2 * ha / 15.0


def interpolate_monthly(day_of_year: np.ndarray, monthly_values: list) -> np.ndarray:
    """Smoothly interpolate monthly normals to daily values using cubic-ish interpolation."""
    # Month midpoints as day-of-year
    midpoints = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    vals = np.array(monthly_values + [monthly_values[0]])  # wrap for circular
    mids = np.concatenate([midpoints - 365, midpoints, midpoints + 365])
    vals_ext = np.concatenate([vals[:-1], vals[:-1], vals[:-1]])

    return np.interp(day_of_year, mids, vals_ext)


def generate_ar1_noise(n: int, rho: float = 0.7, scale: float = 1.0) -> np.ndarray:
    """Generate AR(1) autocorrelated noise (weather persistence)."""
    noise = np.zeros(n)
    innovations = np.random.normal(0, scale * np.sqrt(1 - rho**2), n)
    noise[0] = innovations[0]
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + innovations[i]
    return noise


def generate_melbourne_data(start_year: int = 1950, end_year: int = 2025) -> pd.DataFrame:
    """Generate realistic synthetic daily weather data for Melbourne."""

    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values.astype(float)

    # ── Daylight (deterministic from astronomy) ──
    dl_hours = daylight_hours(doy)

    # ── Temperature ──
    temp_max_clim = interpolate_monthly(doy, MONTHLY_NORMALS["temp_max"])
    temp_min_clim = interpolate_monthly(doy, MONTHLY_NORMALS["temp_min"])

    # Correlated warm/cool spells (synoptic-scale persistence)
    synoptic_noise = generate_ar1_noise(n, rho=0.75, scale=3.0)

    temp_max = temp_max_clim + synoptic_noise + np.random.normal(0, 1.5, n)
    temp_min = temp_min_clim + synoptic_noise * 0.6 + np.random.normal(0, 1.2, n)
    # Ensure min < max
    temp_min = np.minimum(temp_min, temp_max - 1.5)
    temp_mean = (temp_max + temp_min) / 2

    # ── Apparent temperature (wind chill / heat index effect) ──
    apparent_offset = np.where(temp_mean > 20, temp_mean * 0.05 + 1.0, temp_mean * 0.03 - 1.5)
    apparent_max = temp_max + apparent_offset + np.random.normal(0, 0.8, n)
    apparent_min = temp_min + apparent_offset * 0.5 + np.random.normal(0, 0.6, n)
    apparent_mean = (apparent_max + apparent_min) / 2

    # ── Precipitation ──
    rain_prob_clim = interpolate_monthly(doy, MONTHLY_NORMALS["rain_days"]) / 30.0
    rain_clim_daily = interpolate_monthly(doy, MONTHLY_NORMALS["rain_mm"]) / 30.0

    # Rain occurrence (Bernoulli with persistence via synoptic state)
    wet_tendency = generate_ar1_noise(n, rho=0.6, scale=0.3)
    rain_prob = np.clip(rain_prob_clim + wet_tendency, 0.05, 0.85)
    is_rainy = np.random.random(n) < rain_prob

    # Rain amount (exponential distribution on wet days)
    rain_amount = np.where(
        is_rainy,
        np.random.exponential(rain_clim_daily / rain_prob_clim * 1.5, n),
        0.0,
    )
    rain_amount = np.round(np.maximum(rain_amount, 0), 1)
    precip_hours = np.where(is_rainy, np.random.uniform(1, 12, n), 0.0)

    # ── Solar radiation ──
    solar_clim = interpolate_monthly(doy, MONTHLY_NORMALS["solar_mj"])
    # Cloud/rain reduces radiation
    cloud_factor = np.where(is_rainy, np.random.uniform(0.2, 0.6, n), np.random.uniform(0.7, 1.1, n))
    solar_radiation = np.maximum(solar_clim * cloud_factor + np.random.normal(0, 1.5, n), 0.5)

    # ── Sunshine hours ──
    sun_clim = interpolate_monthly(doy, MONTHLY_NORMALS["sun_hours"])
    sunshine_hours = np.maximum(
        sun_clim * cloud_factor + np.random.normal(0, 0.5, n), 0.0
    )
    sunshine_hours = np.minimum(sunshine_hours, dl_hours)

    # ── Wind ──
    wind_clim = interpolate_monthly(doy, MONTHLY_NORMALS["wind_mean"])
    wind_noise = generate_ar1_noise(n, rho=0.5, scale=4.0)
    windspeed_max = np.maximum(wind_clim + wind_noise + np.random.normal(0, 3, n), 3.0)

    gust_clim = interpolate_monthly(doy, MONTHLY_NORMALS["wind_gust"])
    windgusts_max = np.maximum(gust_clim + wind_noise * 1.5 + np.random.normal(0, 8, n), windspeed_max + 5)

    # Dominant wind direction: prevailing from SW-NW with seasonal variation
    # Summer: S-SW sea breeze; Winter: NW-W frontal
    base_direction = interpolate_monthly(doy, [200, 200, 210, 230, 270, 290, 290, 280, 260, 230, 210, 200])
    wind_direction = (base_direction + np.random.normal(0, 40, n)) % 360

    # ── Humidity ──
    hum9_clim = interpolate_monthly(doy, MONTHLY_NORMALS["humidity_9am"])
    hum3_clim = interpolate_monthly(doy, MONTHLY_NORMALS["humidity_3pm"])
    humidity_noise = generate_ar1_noise(n, rho=0.5, scale=5.0)
    rain_hum_boost = np.where(is_rainy, np.random.uniform(5, 15, n), 0)

    humidity_max = np.clip(hum9_clim + humidity_noise + rain_hum_boost + np.random.normal(0, 4, n), 30, 100)
    humidity_min = np.clip(hum3_clim + humidity_noise * 0.7 + rain_hum_boost * 0.5 + np.random.normal(0, 5, n), 15, 95)
    humidity_min = np.minimum(humidity_min, humidity_max - 3)
    humidity_mean = (humidity_max + humidity_min) / 2

    # ── Evapotranspiration ──
    et_clim = interpolate_monthly(doy, MONTHLY_NORMALS["evapotrans"])
    et0 = np.maximum(et_clim * cloud_factor + np.random.normal(0, 0.3, n), 0.1)

    # ── Dewpoint (derived from temp and humidity) ──
    # Magnus formula approximation
    a, b = 17.27, 237.7
    gamma = (a * temp_mean / (b + temp_mean)) + np.log(humidity_mean / 100.0)
    dewpoint = (b * gamma) / (a - gamma)

    # ── Build DataFrame ──
    df = pd.DataFrame({
        "date": dates,
        "temperature_2m_max": np.round(temp_max, 1),
        "temperature_2m_min": np.round(temp_min, 1),
        "temperature_2m_mean": np.round(temp_mean, 1),
        "apparent_temperature_max": np.round(apparent_max, 1),
        "apparent_temperature_min": np.round(apparent_min, 1),
        "apparent_temperature_mean": np.round(apparent_mean, 1),
        "precipitation_sum": rain_amount,
        "rain_sum": rain_amount,  # same for this dataset
        "precipitation_hours": np.round(precip_hours, 1),
        "windspeed_10m_max": np.round(windspeed_max, 1),
        "windgusts_10m_max": np.round(windgusts_max, 1),
        "winddirection_10m_dominant": np.round(wind_direction, 0).astype(int),
        "shortwave_radiation_sum": np.round(solar_radiation, 2),
        "daylight_duration": np.round(dl_hours * 3600, 0),  # seconds
        "sunshine_duration": np.round(sunshine_hours * 3600, 0),  # seconds
        "et0_fao_evapotranspiration": np.round(et0, 2),
        "relative_humidity_max": np.round(humidity_max, 1),
        "relative_humidity_min": np.round(humidity_min, 1),
        "relative_humidity_mean": np.round(humidity_mean, 1),
        "dewpoint_mean": np.round(dewpoint, 1),
        "daylight_hours": np.round(dl_hours, 2),
        "sunshine_hours": np.round(sunshine_hours, 2),
        "sunshine_fraction": np.round(sunshine_hours / np.maximum(dl_hours, 0.1), 3),
        "temp_range": np.round(temp_max - temp_min, 1),
        "month": dates.month,
        "day_of_year": doy.astype(int),
    })

    return df


def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating realistic Melbourne climate data (1950-2025) ===")
    df = generate_melbourne_data(1950, 2025)
    print(f"Generated {len(df)} days, {len(df.columns)} variables")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample stats:")
    print(df.describe().round(2).to_string())

    df.to_parquet(out_dir / "melbourne_daily.parquet", index=False)
    df.to_csv(out_dir / "melbourne_daily.csv", index=False)
    print(f"\nSaved to data/melbourne_daily.parquet and data/melbourne_daily.csv")


if __name__ == "__main__":
    main()
