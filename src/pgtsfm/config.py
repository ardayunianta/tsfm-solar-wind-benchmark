"""Shared settings: study period, splits, series definitions and weather points."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
# PGTSFM_RESULTS points the scripts at another results folder (17_timing.py uses a temporary one)
RESULTS = Path(os.environ.get("PGTSFM_RESULTS", ROOT / "results"))

# Day-ahead weather archives (Open-Meteo previous runs) start in March 2024.
START = "2024-03-15"
END = "2026-09-30"  # inclusive

SPLITS = {
    "train": ("2024-04-01", "2025-03-31"),
    "val": ("2025-04-01", "2025-09-30"),
    "test": ("2025-10-01", "2026-09-30"),
}

# Forecasts for day D are issued at 09:00 UTC on day D-1; day D runs 00:00-23:45 UTC.
ISSUE_HOUR_UTC = 9
STEPS_PER_DAY = 96  # 15-minute resolution
CONTEXT_DAYS = 7

QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Solar: Elia publishes one series per province. Weather is taken at the province centroid.
SOLAR_REGIONS = {
    "Antwerp": [(51.22, 4.65)],
    "Brussels": [(50.85, 4.35)],
    "East-Flanders": [(51.03, 3.82)],
    "Flemish-Brabant": [(50.88, 4.60)],
    "Hainaut": [(50.45, 3.95)],
    "Limburg": [(50.98, 5.40)],
    "Liège": [(50.52, 5.65)],
    "Luxembourg": [(49.95, 5.45)],
    "Namur": [(50.30, 4.85)],
    "Walloon-Brabant": [(50.67, 4.60)],
    "West-Flanders": [(51.03, 3.05)],
}

# Wind: Elia publishes offshore plus onshore Flanders/Wallonia split by grid connection.
# Onshore series cover whole regions, so weather is averaged over several points.
FLANDERS_POINTS = [(51.30, 4.30), (51.10, 3.75), (51.00, 3.00), (51.00, 5.40)]
WALLONIA_POINTS = [(50.45, 3.95), (50.30, 4.85), (50.50, 5.60), (50.00, 5.40)]
WIND_SERIES = {
    "Offshore": {"offshoreonshore": "Offshore", "region": "Federal", "gridconnectiontype": "Elia",
                 "points": [(51.62, 2.93)]},
    "Onshore-Flanders-Elia": {"offshoreonshore": "Onshore", "region": "Flanders", "gridconnectiontype": "Elia",
                              "points": FLANDERS_POINTS},
    "Onshore-Flanders-DSO": {"offshoreonshore": "Onshore", "region": "Flanders", "gridconnectiontype": "Dso",
                             "points": FLANDERS_POINTS},
    "Onshore-Wallonia-Elia": {"offshoreonshore": "Onshore", "region": "Wallonia", "gridconnectiontype": "Elia",
                              "points": WALLONIA_POINTS},
    "Onshore-Wallonia-DSO": {"offshoreonshore": "Onshore", "region": "Wallonia", "gridconnectiontype": "Dso",
                             "points": WALLONIA_POINTS},
}

# ECMWF IFS 0.25 deg forecasts from Open-Meteo. "previous_day2" = the value forecast about
# 48 h before the valid time, which was available before our 09:00 UTC issue time on D-1.
NWP_MODEL = "ecmwf_ifs025"
NWP_VARS = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
    "temperature_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_100m",
]
NWP_LAGS = ["previous_day2", "previous_day1"]  # day2 = main (no leakage); day1 = optimistic check
