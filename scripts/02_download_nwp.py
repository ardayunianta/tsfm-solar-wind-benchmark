"""Download archived day-ahead weather forecasts (ECMWF IFS) for every weather point.

Uses the free Open-Meteo Previous Runs API. Each variable is requested with the suffix
"_previous_day2" (value forecast ~48 h before valid time) and "_previous_day1".
Output: data/raw/nwp_<lat>_<lon>.parquet, hourly, UTC.

Note: Open-Meteo radiation values are averages over the preceding hour; build_dataset.py
shifts them by -30 min before interpolating to 15 minutes.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import END, NWP_LAGS, NWP_MODEL, NWP_VARS, RAW, SOLAR_REGIONS, START, WIND_SERIES  # noqa: E402

API = "https://previous-runs-api.open-meteo.com/v1/forecast"


def all_points():
    pts = {p for pts in SOLAR_REGIONS.values() for p in pts}
    pts |= {p for s in WIND_SERIES.values() for p in s["points"]}
    return sorted(pts)


def fetch(lat, lon, model):
    hourly = [f"{v}_{lag}" for v in NWP_VARS for lag in NWP_LAGS]
    frames = []
    # Request one year at a time to stay within API limits.
    for y0, y1 in [(START, "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", END)]:
        params = {"latitude": lat, "longitude": lon, "models": model, "hourly": ",".join(hourly),
                  "start_date": y0, "end_date": y1, "timezone": "GMT"}
        for attempt in range(6):
            r = requests.get(API, params=params, timeout=300)
            if r.ok:
                break
            print(f"    retry {attempt}: {r.status_code} {r.text[:120]}", flush=True)
            time.sleep(30 * (attempt + 1))
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        frames.append(df)
        time.sleep(2)
    return pd.concat(frames, ignore_index=True)


def nwp_path(lat, lon, model=NWP_MODEL):
    # The main model (ECMWF) keeps the original file names; extra models carry their name.
    tag = "" if model == NWP_MODEL else f"{model}_"
    return RAW / f"nwp_{tag}{lat:.2f}_{lon:.2f}.parquet"


def main(model=NWP_MODEL):
    RAW.mkdir(parents=True, exist_ok=True)
    for lat, lon in all_points():
        path = nwp_path(lat, lon, model)
        if path.exists():
            print(f"{path.name} exists, skipping")
            continue
        df = fetch(lat, lon, model)
        df.to_parquet(path, index=False)
        miss = df.drop(columns="time").isna().mean().round(3).to_dict()
        print(f"Saved {path.name}: {len(df):,} hours; missing share by column: {miss}", flush=True)


if __name__ == "__main__":
    # python scripts/02_download_nwp.py [model]   e.g. icon_eu, gfs_seamless
    main(sys.argv[1] if len(sys.argv) > 1 else NWP_MODEL)
