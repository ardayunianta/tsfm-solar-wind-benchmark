"""Build one tidy 15-minute table per technology (solar, wind).

Columns: series, ds (UTC), y (load factor = measured / monitored capacity), capacity,
Elia day-ahead forecasts as load factors (elia_da11, elia_da11_p10, elia_da11_p90),
weather forecasts for each lag (e.g. ghi_d2, ws100_d2, ...), and solar geometry.
Output: data/processed/{solar,wind}.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import END, NWP_LAGS, PROCESSED, RAW, SOLAR_REGIONS, START, WIND_SERIES  # noqa: E402

SHORT = {"shortwave_radiation": "ghi", "direct_radiation": "dir", "diffuse_radiation": "dif",
         "cloud_cover": "cc", "temperature_2m": "t2m", "wind_speed_10m": "ws10",
         "wind_speed_100m": "ws100", "wind_direction_100m": "wd100"}
RADIATION = {"ghi", "dir", "dif"}
INDEX = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(hours=23, minutes=45), freq="15min", tz="UTC")


# Extra weather models: file tag and column prefix. ECMWF (the main model) has no prefix.
EXTRA_NWP = {"icon_eu": "icon_", "gfs_seamless": "gfs_"}


def load_nwp(points, model_tag="", prefix=""):
    """Average hourly NWP over points, then put it on the 15-minute grid."""
    frames = []
    for lat, lon in points:
        df = pd.read_parquet(RAW / f"nwp_{model_tag}{lat:.2f}_{lon:.2f}.parquet").set_index("time")
        frames.append(df)
    cols, radiation = {}, set()
    for lag in NWP_LAGS:
        tag = "d" + lag[-1]
        for long, short in SHORT.items():
            stack = np.stack([f[f"{long}_{lag}"].to_numpy(dtype=float) for f in frames])
            if short == "wd100":
                # Average directions as unit vectors.
                rad = np.deg2rad(stack)
                cols[f"{prefix}wd100_sin_{tag}"] = np.nanmean(np.sin(rad), axis=0)
                cols[f"{prefix}wd100_cos_{tag}"] = np.nanmean(np.cos(rad), axis=0)
            else:
                cols[f"{prefix}{short}_{tag}"] = np.nanmean(stack, axis=0)
                if short in RADIATION:
                    radiation.add(f"{prefix}{short}_{tag}")
    hourly = pd.DataFrame(cols, index=frames[0].index)
    out = {}
    for c in hourly.columns:
        s = hourly[c]
        if c in radiation:
            s = s.copy()
            s.index = s.index - pd.Timedelta(minutes=30)  # preceding-hour mean -> mid-interval
        out[c] = s.reindex(s.index.union(INDEX)).interpolate("time", limit=8).reindex(INDEX)
    nwp = pd.DataFrame(out)
    for c in radiation:
        nwp[c] = nwp[c].clip(lower=0)
    return nwp


def all_nwp(points):
    parts = [load_nwp(points)]
    parts += [load_nwp(points, f"{model}_", prefix) for model, prefix in EXTRA_NWP.items()]
    return pd.concat(parts, axis=1)


def solar_geometry(points):
    lat = np.mean([p[0] for p in points])
    lon = np.mean([p[1] for p in points])
    loc = pvlib.location.Location(lat, lon, tz="UTC")
    # Evaluate at the middle of each 15-minute interval.
    mid = INDEX + pd.Timedelta(minutes=7.5)
    sp = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen")
    geo = pd.DataFrame({"zenith": sp["apparent_zenith"].to_numpy(),
                        "cos_zenith": np.cos(np.deg2rad(sp["apparent_zenith"].to_numpy())).clip(0),
                        "ghi_clear": cs["ghi"].to_numpy()}, index=INDEX)
    return geo


def elia_frame(df):
    df = df[df["resolutioncode"] == "PT15M"].copy()
    cols = ["measured", "monitoredcapacity", "dayahead11hforecast", "dayahead11hconfidence10", "dayahead11hconfidence90"]
    # min_count=1 keeps a timestamp missing when all its values are missing (plain sum would give 0).
    df = df.groupby("datetime")[cols].sum(min_count=1)
    df = df.reindex(INDEX)
    cap = df["monitoredcapacity"].replace(0, np.nan)
    return pd.DataFrame({
        "y": (df["measured"] / cap).clip(lower=0),
        "capacity": cap,
        "elia_da11": (df["dayahead11hforecast"] / cap).clip(lower=0),
        "elia_da11_p10": (df["dayahead11hconfidence10"] / cap).clip(lower=0),
        "elia_da11_p90": (df["dayahead11hconfidence90"] / cap).clip(lower=0),
    }, index=INDEX)


def build(tech):
    raw = pd.read_parquet(RAW / f"elia_{tech}.parquet")
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    parts = []
    if tech == "solar":
        series = {name: (raw[raw["region"] == name], pts) for name, pts in SOLAR_REGIONS.items()}
    else:
        series = {}
        for name, spec in WIND_SERIES.items():
            m = ((raw["offshoreonshore"] == spec["offshoreonshore"]) & (raw["region"] == spec["region"]) &
                 (raw["gridconnectiontype"] == spec["gridconnectiontype"]))
            series[name] = (raw[m], spec["points"])
    for name, (sub, pts) in series.items():
        df = elia_frame(sub).join(all_nwp(pts)).join(solar_geometry(pts))
        df.insert(0, "series", name)
        df.index.name = "ds"
        parts.append(df.reset_index())
        miss = df[["y", "elia_da11", "ghi_d2", "ws100_d2"]].isna().mean().round(4).to_dict()
        print(f"  {tech}/{name}: {len(df):,} rows, missing share {miss}")
    out = pd.concat(parts, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED / f"{tech}.parquet", index=False)
    print(f"Saved {tech}.parquet: {out.shape}")


if __name__ == "__main__":
    for tech in ("solar", "wind"):
        build(tech)
