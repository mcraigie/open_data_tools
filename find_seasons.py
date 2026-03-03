#!/usr/bin/env python3
"""
Melbourne Data-Driven Season Discovery

Takes daily weather data and discovers natural contiguous "seasons" using
changepoint detection on the annual climate cycle.

Approach:
1. Average each day-of-year across all years → 366-row "typical year" profile
2. Standardize all variables
3. Use Ruptures (kernel changepoint detection) to find optimal breakpoints
   in this circular time series — these breakpoints define season boundaries
4. Evaluate multiple candidate numbers of seasons (2-12) using BIC / elbow
5. Characterize and visualize the discovered seasons

Key constraint: seasons must be CONTIGUOUS blocks of days (no scattered days).
This is naturally enforced by changepoint detection on the day-of-year axis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import ruptures as rpt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────
# Variables to use for clustering (subset of what's in the data)
CLUSTER_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "temp_range",
    "apparent_temperature_mean",
    "precipitation_sum",
    "precipitation_hours",
    "shortwave_radiation_sum",
    "sunshine_hours",
    "sunshine_fraction",
    "daylight_hours",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "et0_fao_evapotranspiration",
    "relative_humidity_mean",
    "dewpoint_mean",
]

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")

# Candidate range for number of seasons
MIN_SEASONS = 2
MAX_SEASONS = 12


def load_data() -> pd.DataFrame:
    """Load the daily Melbourne weather data."""
    parquet_path = DATA_DIR / "melbourne_daily.parquet"
    csv_path = DATA_DIR / "melbourne_daily.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        raise FileNotFoundError(
            "No data found. Run fetch_data.py or generate_data.py first."
        )

    # Ensure day_of_year exists
    if "day_of_year" not in df.columns:
        df["day_of_year"] = df["date"].dt.dayofyear

    return df


def build_annual_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average each day-of-year (1-366) across all years to produce
    a single "typical year" climate profile.
    """
    available = [v for v in CLUSTER_VARIABLES if v in df.columns]
    print(f"  Using {len(available)} variables: {available}")

    profile = df.groupby("day_of_year")[available].mean()

    # Smooth with Savitzky-Golay to reduce noise while preserving shape
    for col in available:
        vals = profile[col].values
        # Circular padding for smooth wrap-around
        padded = np.concatenate([vals[-30:], vals, vals[:30]])
        smoothed = savgol_filter(padded, window_length=15, polyorder=3)
        profile[col] = smoothed[30:-30]

    return profile


def find_changepoints(profile: pd.DataFrame, n_seasons: int) -> list[int]:
    """
    Find changepoints (season boundaries) using kernel changepoint detection.
    Returns sorted list of day-of-year breakpoints.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(profile.values)

    # Use dynamic programming with RBF kernel for optimal segmentation
    algo = rpt.KernelCPD(kernel="rbf", min_size=15).fit(X)
    breakpoints = algo.predict(n_bkps=n_seasons - 1)

    # breakpoints includes the final index (366); remove it, keep only interior breaks
    breakpoints = [b for b in breakpoints if b < len(profile)]
    return sorted(breakpoints)


def compute_segmentation_cost(profile: pd.DataFrame, n_seasons: int) -> tuple[float, list[int]]:
    """
    Compute the total residual cost for a given number of seasons.
    Returns (cost, breakpoints).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(profile.values)

    algo = rpt.KernelCPD(kernel="rbf", min_size=15).fit(X)
    breakpoints = algo.predict(n_bkps=n_seasons - 1)

    # Compute within-segment variance as cost
    interior_bkps = sorted([b for b in breakpoints if b < len(profile)])
    boundaries = [0] + interior_bkps + [len(profile)]

    total_cost = 0.0
    for i in range(len(boundaries) - 1):
        segment = X[boundaries[i]:boundaries[i+1]]
        total_cost += np.sum(np.var(segment, axis=0)) * len(segment)

    return total_cost, interior_bkps


def find_optimal_seasons(profile: pd.DataFrame) -> tuple[int, dict]:
    """
    Evaluate different numbers of seasons and find the most compelling breakdown.

    Uses a penalized cost approach:
    - Compute segmentation cost for K=2..12
    - Use BIC-like penalty to avoid over-segmentation
    - Also compute "improvement ratio" to find the elbow
    """
    results = {}
    n_days = len(profile)
    n_vars = profile.shape[1]

    print("\n  Evaluating candidate season counts:")
    for k in range(MIN_SEASONS, MAX_SEASONS + 1):
        cost, bkps = compute_segmentation_cost(profile, k)
        # BIC-like penalty: penalize for number of parameters
        n_params = k * n_vars  # each segment has n_vars means
        bic = cost + n_params * np.log(n_days)
        results[k] = {"cost": cost, "bic": bic, "breakpoints": bkps}
        print(f"    K={k:2d}: cost={cost:8.1f}  BIC={bic:8.1f}  breakpoints={bkps}")

    # Find the K that minimizes BIC
    best_k = min(results, key=lambda k: results[k]["bic"])

    # Also compute improvement ratios for elbow detection
    costs = [results[k]["cost"] for k in sorted(results)]
    improvements = []
    for i in range(1, len(costs)):
        if costs[i-1] > 0:
            improvements.append((costs[i-1] - costs[i]) / costs[i-1])
        else:
            improvements.append(0)

    # Find elbow: where marginal improvement drops below threshold
    elbow_k = best_k
    for i, imp in enumerate(improvements):
        if i > 0 and imp < 0.03:  # less than 3% improvement
            elbow_k = i + MIN_SEASONS
            break

    # Use BIC winner but report both
    print(f"\n  BIC-optimal: {best_k} seasons")
    print(f"  Elbow method: {elbow_k} seasons")

    return best_k, results


def assign_seasons(profile: pd.DataFrame, breakpoints: list[int], n_seasons: int) -> np.ndarray:
    """
    Assign each day-of-year to a season based on breakpoints.
    Returns array of season labels (0-indexed).
    """
    labels = np.zeros(len(profile), dtype=int)
    boundaries = sorted(breakpoints)

    for i, doy_idx in enumerate(range(len(profile))):
        season = 0
        for b in boundaries:
            if doy_idx >= b:
                season += 1
        labels[i] = season

    return labels


def characterize_seasons(
    profile: pd.DataFrame, labels: np.ndarray, breakpoints: list[int]
) -> pd.DataFrame:
    """
    Produce a summary table characterizing each discovered season.
    """
    profile_with_labels = profile.copy()
    profile_with_labels["season"] = labels

    # Build boundaries as date ranges
    boundaries = [0] + sorted(breakpoints) + [366]

    season_info = []
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    for s in range(len(boundaries) - 1):
        start_doy = boundaries[s] + 1  # 1-indexed
        end_doy = boundaries[s + 1]    # inclusive

        # Convert DOY to approximate date
        start_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=start_doy - 1)
        end_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=end_doy - 1)

        segment = profile_with_labels[profile_with_labels["season"] == s]
        n_days = len(segment)

        info = {
            "season": s + 1,
            "start_doy": start_doy,
            "end_doy": end_doy,
            "start_date": start_date.strftime("%d %b"),
            "end_date": end_date.strftime("%d %b"),
            "n_days": n_days,
            "mean_temp": segment["temperature_2m_mean"].mean() if "temperature_2m_mean" in segment else None,
            "mean_max_temp": segment["temperature_2m_max"].mean() if "temperature_2m_max" in segment else None,
            "mean_min_temp": segment["temperature_2m_min"].mean() if "temperature_2m_min" in segment else None,
            "mean_rain_mm": segment["precipitation_sum"].mean() if "precipitation_sum" in segment else None,
            "mean_sunshine_hrs": segment["sunshine_hours"].mean() if "sunshine_hours" in segment else None,
            "mean_daylight_hrs": segment["daylight_hours"].mean() if "daylight_hours" in segment else None,
            "mean_wind_max": segment["windspeed_10m_max"].mean() if "windspeed_10m_max" in segment else None,
            "mean_humidity": segment["relative_humidity_mean"].mean() if "relative_humidity_mean" in segment else None,
            "mean_solar_mj": segment["shortwave_radiation_sum"].mean() if "shortwave_radiation_sum" in segment else None,
        }
        season_info.append(info)

    return pd.DataFrame(season_info)


def name_seasons(char_df: pd.DataFrame) -> list[str]:
    """
    Auto-generate distinctive names for each season based on what makes it
    stand out relative to the other seasons (not absolute thresholds).
    """
    names = []
    n = len(char_df)

    # Rank each season on each dimension (0 = lowest, n-1 = highest)
    def rank_col(col):
        vals = char_df[col].values
        order = np.argsort(np.argsort(vals))  # rank 0..n-1
        return order

    temp_rank = rank_col("mean_temp") if "mean_temp" in char_df else np.zeros(n)
    rain_rank = rank_col("mean_rain_mm") if "mean_rain_mm" in char_df else np.zeros(n)
    sun_rank = rank_col("mean_sunshine_hrs") if "mean_sunshine_hrs" in char_df else np.zeros(n)
    wind_rank = rank_col("mean_wind_max") if "mean_wind_max" in char_df else np.zeros(n)
    humidity_rank = rank_col("mean_humidity") if "mean_humidity" in char_df else np.zeros(n)

    temp_labels = {0: "Coldest", 1: "Cold", n-2: "Warm", n-1: "Hottest"}
    mid_temp = {i: ["Cool", "Mild", "Mild-Cool", "Mild-Warm"][i % 4] for i in range(2, n-2)}
    temp_labels.update(mid_temp)

    for i in range(n):
        parts = []
        tr = int(temp_rank[i])

        # Temperature - always include, use relative labels
        if n <= 4:
            t_names = {0: "Cold", 1: "Cool", 2: "Warm", 3: "Hot"}
            parts.append(t_names.get(tr, "Mild"))
        else:
            parts.append(temp_labels.get(tr, "Mild"))

        # Pick the most distinctive non-temperature feature
        distinguishers = []
        sr = int(sun_rank[i])
        rr = int(rain_rank[i])
        wr = int(wind_rank[i])
        hr = int(humidity_rank[i])

        if sr == n - 1:
            distinguishers.append("Bright")
        elif sr == 0:
            distinguishers.append("Dark")
        elif sr >= n - 2:
            distinguishers.append("Sunny")
        elif sr <= 1:
            distinguishers.append("Grey")

        if rr == n - 1:
            distinguishers.append("Wettest")
        elif rr == 0:
            distinguishers.append("Driest")
        elif rr >= n - 2:
            distinguishers.append("Wet")

        if wr >= n - 2:
            distinguishers.append("Blustery")
        elif wr == 0:
            distinguishers.append("Calm")

        if hr >= n - 2 and "Wet" not in str(distinguishers):
            distinguishers.append("Humid")
        elif hr == 0 and "Dry" not in str(distinguishers):
            distinguishers.append("Crisp")

        # Take up to 2 distinguishers
        parts.extend(distinguishers[:2])

        names.append(", ".join(parts))

    return names


# ── Visualization ────────────────────────────────────────────────────────

def plot_cost_curve(results: dict, best_k: int):
    """Plot BIC curve for number of seasons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ks = sorted(results.keys())
    costs = [results[k]["cost"] for k in ks]
    bics = [results[k]["bic"] for k in ks]

    ax1.plot(ks, costs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.axvline(best_k, color="#F44336", linestyle="--", alpha=0.7, label=f"Selected: {best_k}")
    ax1.set_xlabel("Number of Seasons", fontsize=12)
    ax1.set_ylabel("Segmentation Cost", fontsize=12)
    ax1.set_title("Within-Segment Variance", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, bics, "o-", color="#4CAF50", linewidth=2, markersize=8)
    ax2.axvline(best_k, color="#F44336", linestyle="--", alpha=0.7, label=f"Optimal: {best_k}")
    ax2.set_xlabel("Number of Seasons", fontsize=12)
    ax2.set_ylabel("BIC Score", fontsize=12)
    ax2.set_title("Bayesian Information Criterion", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "season_count_selection.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_annual_profile_with_seasons(
    profile: pd.DataFrame, labels: np.ndarray, breakpoints: list[int],
    char_df: pd.DataFrame, season_names: list[str]
):
    """Plot the annual profile colored by discovered seasons."""
    n_seasons = len(char_df)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_seasons, 3)))

    # Key variables to display
    plot_vars = [
        ("temperature_2m_mean", "Mean Temperature (°C)"),
        ("precipitation_sum", "Mean Daily Rainfall (mm)"),
        ("sunshine_hours", "Sunshine Hours"),
        ("relative_humidity_mean", "Relative Humidity (%)"),
        ("windspeed_10m_max", "Max Wind Speed (km/h)"),
        ("shortwave_radiation_sum", "Solar Radiation (MJ/m²)"),
    ]
    plot_vars = [(v, l) for v, l in plot_vars if v in profile.columns]

    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(16, 3.5 * len(plot_vars)), sharex=True)
    if len(plot_vars) == 1:
        axes = [axes]

    doy = profile.index.values

    for ax, (var, ylabel) in zip(axes, plot_vars):
        vals = profile[var].values

        # Plot each season segment in its color
        boundaries = [0] + sorted(breakpoints) + [len(profile)]
        for s in range(len(boundaries) - 1):
            start = boundaries[s]
            end = boundaries[s + 1]
            mask = slice(start, end)
            ax.fill_between(
                doy[mask], vals[mask], alpha=0.3, color=colors[s],
                label=f"Season {s+1}: {season_names[s]}" if ax == axes[0] else None
            )
            ax.plot(doy[mask], vals[mask], color=colors[s], linewidth=1.5)

        # Draw breakpoint lines
        for bp in breakpoints:
            ax.axvline(bp, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.2)

    # X-axis as months
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[-1].set_xticks(month_starts)
    axes[-1].set_xticklabels(month_labels, fontsize=11)
    axes[-1].set_xlabel("Day of Year", fontsize=12)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.35),
                   ncol=min(n_seasons, 4), fontsize=10, frameon=True)

    fig.suptitle(
        f"Melbourne's {n_seasons} Data-Driven Seasons",
        fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "season_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_season_calendar(labels: np.ndarray, breakpoints: list[int],
                         season_names: list[str], char_df: pd.DataFrame):
    """Plot a circular calendar showing when each season falls."""
    n_seasons = len(char_df)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_seasons, 3)))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Convert day-of-year to radians (Jan 1 = top)
    boundaries = [0] + sorted(breakpoints) + [366]

    for s in range(len(boundaries) - 1):
        start_angle = (boundaries[s] / 366) * 2 * np.pi - np.pi / 2
        end_angle = (boundaries[s + 1] / 366) * 2 * np.pi - np.pi / 2
        theta = np.linspace(start_angle, end_angle, 100)

        ax.fill_between(theta, 0.5, 1.0, color=colors[s], alpha=0.6)
        ax.fill_between(theta, 0.3, 0.5, color=colors[s], alpha=0.3)

        # Label at midpoint
        mid_angle = (start_angle + end_angle) / 2
        t = char_df.iloc[s]["mean_temp"]
        temp_str = f"{t:.0f}°C" if pd.notna(t) else ""
        ax.text(
            mid_angle, 0.75,
            f"Season {s+1}\n{season_names[s]}\n{temp_str}",
            ha="center", va="center", fontsize=8, fontweight="bold"
        )

    # Month labels on outside
    month_angles = [(m / 366) * 2 * np.pi - np.pi / 2 for m in
                    [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for angle, label in zip(month_angles, month_labels):
        ax.text(angle, 1.12, label, ha="center", va="center", fontsize=10)

    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Melbourne Season Calendar\n(Data-Driven)", fontsize=14,
                 fontweight="bold", pad=20)

    plt.savefig(OUTPUT_DIR / "season_calendar.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_comparison(char_df: pd.DataFrame, season_names: list[str]):
    """Radar/spider chart comparing seasons across all dimensions."""
    n_seasons = len(char_df)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_seasons, 3)))

    metrics = [
        ("mean_temp", "Temperature"),
        ("mean_rain_mm", "Rainfall"),
        ("mean_sunshine_hrs", "Sunshine"),
        ("mean_humidity", "Humidity"),
        ("mean_wind_max", "Wind"),
        ("mean_solar_mj", "Solar Rad."),
    ]
    metrics = [(col, label) for col, label in metrics if col in char_df.columns and char_df[col].notna().all()]

    if len(metrics) < 3:
        return

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    for s in range(n_seasons):
        values = []
        for col, _ in metrics:
            v = char_df.iloc[s][col]
            # Normalize 0-1 within the column
            col_min = char_df[col].min()
            col_max = char_df[col].max()
            if col_max > col_min:
                values.append((v - col_min) / (col_max - col_min))
            else:
                values.append(0.5)
        values += values[:1]

        ax.plot(angles, values, "o-", color=colors[s], linewidth=2,
                label=f"S{s+1}: {season_names[s]}", markersize=6)
        ax.fill(angles, values, color=colors[s], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l for _, l in metrics], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Season Comparison (Normalized)", fontsize=14, fontweight="bold", pad=20)

    plt.savefig(OUTPUT_DIR / "season_radar.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(profile: pd.DataFrame, labels: np.ndarray, breakpoints: list[int]):
    """Heatmap of all normalized variables across the year, with season boundaries."""
    scaler = StandardScaler()
    data = scaler.fit_transform(profile.values)

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(data.T, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")

    # Season boundary lines
    for bp in breakpoints:
        ax.axvline(bp, color="black", linewidth=2, linestyle="--")

    ax.set_yticks(range(len(profile.columns)))
    ax.set_yticklabels(profile.columns, fontsize=9)

    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=11)

    plt.colorbar(im, ax=ax, label="Standardized Value", shrink=0.8)
    ax.set_title("Melbourne Annual Climate Heatmap with Season Boundaries",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Day of Year", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "climate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MELBOURNE DATA-DRIVEN SEASON DISCOVERY")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} daily records")

    # 2. Build annual profile
    print("\n[2/6] Building annual climate profile (averaging across years)...")
    profile = build_annual_profile(df)
    print(f"  Profile shape: {profile.shape} (days × variables)")

    # 3. Find optimal number of seasons
    print("\n[3/6] Finding optimal number of seasons...")
    best_k, results = find_optimal_seasons(profile)

    # 4. Get the breakpoints for the optimal K
    print(f"\n[4/6] Extracting {best_k} seasons...")
    breakpoints = results[best_k]["breakpoints"]
    labels = assign_seasons(profile, breakpoints, best_k)
    print(f"  Breakpoints (day-of-year): {breakpoints}")

    # Convert breakpoints to dates for display
    for bp in breakpoints:
        bp_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=bp)
        print(f"    Day {bp:3d} → {bp_date.strftime('%d %B')}")

    # 5. Characterize seasons
    print(f"\n[5/6] Characterizing seasons...")
    char_df = characterize_seasons(profile, labels, breakpoints)
    season_names = name_seasons(char_df)
    char_df["name"] = season_names

    print("\n" + "=" * 70)
    print(f"  DISCOVERED {best_k} SEASONS FOR MELBOURNE")
    print("=" * 70)
    for _, row in char_df.iterrows():
        s = int(row["season"])
        print(f"\n  Season {s}: {season_names[s-1]}")
        print(f"    Period:      {row['start_date']} – {row['end_date']} ({int(row['n_days'])} days)")
        if pd.notna(row.get("mean_temp")):
            print(f"    Temperature: {row['mean_temp']:.1f}°C avg  (max {row['mean_max_temp']:.1f}°C, min {row['mean_min_temp']:.1f}°C)")
        if pd.notna(row.get("mean_rain_mm")):
            print(f"    Rainfall:    {row['mean_rain_mm']:.1f} mm/day avg")
        if pd.notna(row.get("mean_sunshine_hrs")):
            print(f"    Sunshine:    {row['mean_sunshine_hrs']:.1f} hrs/day avg")
        if pd.notna(row.get("mean_humidity")):
            print(f"    Humidity:    {row['mean_humidity']:.0f}% avg")
        if pd.notna(row.get("mean_wind_max")):
            print(f"    Max wind:    {row['mean_wind_max']:.0f} km/h avg")
        if pd.notna(row.get("mean_solar_mj")):
            print(f"    Solar:       {row['mean_solar_mj']:.1f} MJ/m²/day avg")

    # Save characterization
    char_df.to_csv(OUTPUT_DIR / "season_summary.csv", index=False)
    print(f"\n  Season summary saved to {OUTPUT_DIR / 'season_summary.csv'}")

    # 6. Generate visualizations
    print(f"\n[6/6] Generating visualizations...")
    plot_cost_curve(results, best_k)
    print("  ✓ season_count_selection.png")

    plot_annual_profile_with_seasons(profile, labels, breakpoints, char_df, season_names)
    print("  ✓ season_profiles.png")

    plot_season_calendar(labels, breakpoints, season_names, char_df)
    print("  ✓ season_calendar.png")

    plot_radar_comparison(char_df, season_names)
    print("  ✓ season_radar.png")

    plot_heatmap(profile, labels, breakpoints)
    print("  ✓ climate_heatmap.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
