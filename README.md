# Melbourne Data-Driven Season Discovery

Discovers Melbourne's natural "seasons" from multi-dimensional weather data using changepoint detection — no preconceptions about how many seasons exist or when they start.

## How it works

1. **Data**: 75 years (1950–2025) of daily weather from [Open-Meteo](https://open-meteo.com/) — temperature, rainfall, sunshine, solar radiation, humidity, wind, daylight, evapotranspiration
2. **Annual profile**: Averages each day-of-year across all years to get one "typical year" (366 data points × 16 variables), smoothed with Savitzky-Golay filter
3. **Changepoint detection**: Uses kernel-based changepoint detection ([ruptures](https://centre-borelli.github.io/ruptures-docs/)) to find natural breakpoints in the multivariate annual cycle
4. **Optimal K selection**: Evaluates 2–12 seasons using BIC to find the most parsimonious segmentation
5. **Characterization**: Names and describes each season by its climate fingerprint

Seasons are guaranteed to be **contiguous blocks of days** — this falls out naturally from changepoint detection on the day-of-year axis (no post-hoc merging needed).

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch real data (requires internet to Open-Meteo — free, no API key)
python fetch_data.py

# Step 2: Run season discovery
python find_seasons.py
```

If you can't reach Open-Meteo, you can use the synthetic data generator instead:
```bash
python generate_data.py   # generates realistic synthetic data based on BOM normals
python find_seasons.py
```

## Output

All results go to `output/`:

| File | Description |
|------|-------------|
| `season_summary.csv` | Table of all discovered seasons with date ranges and climate stats |
| `season_profiles.png` | Annual profiles of key variables, colored by season |
| `season_calendar.png` | Circular calendar showing when each season falls |
| `season_radar.png` | Radar chart comparing seasons across all dimensions |
| `climate_heatmap.png` | Full heatmap of all variables across the year |
| `season_count_selection.png` | BIC/cost curves showing why K seasons was chosen |

## Variables used

| Variable | Description |
|----------|-------------|
| `temperature_2m_mean/max/min` | Air temperature at 2m (°C) |
| `apparent_temperature_mean` | Feels-like temperature (°C) |
| `temp_range` | Daily temperature range (°C) |
| `precipitation_sum` | Daily rainfall (mm) |
| `precipitation_hours` | Hours with precipitation |
| `shortwave_radiation_sum` | Total solar energy (MJ/m²) |
| `sunshine_hours` | Actual sunshine hours |
| `sunshine_fraction` | Sunshine / daylight ratio |
| `daylight_hours` | Astronomical daylight hours |
| `windspeed_10m_max` | Max wind speed (km/h) |
| `windgusts_10m_max` | Max wind gust (km/h) |
| `et0_fao_evapotranspiration` | Reference evapotranspiration (mm) |
| `relative_humidity_mean` | Mean relative humidity (%) |
| `dewpoint_mean` | Mean dewpoint temperature (°C) |

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, requests, ruptures, pyarrow
