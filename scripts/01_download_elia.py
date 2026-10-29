"""Download Elia solar, wind and imbalance-price data for the study period.

Elia Open Data (opendata.elia.be) is public. The "exports" endpoint returns the whole
filtered table in one file, so no paging is needed. Output: data/raw/elia_*.parquet
"""
import io
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import END, RAW, START  # noqa: E402

API = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{ds}/exports/parquet"

FORECAST_FIELDS = ("datetime,resolutioncode,region,measured,monitoredcapacity,"
                   "dayahead11hforecast,dayahead11hconfidence10,dayahead11hconfidence90,"
                   "dayaheadforecast,dayaheadconfidence10,dayaheadconfidence90")

JOBS = {
    "elia_solar": ("ods032", FORECAST_FIELDS),
    "elia_wind": ("ods031", FORECAST_FIELDS + ",offshoreonshore,gridconnectiontype"),
    "elia_imbalance": ("ods134", None),
}


def download(ds, select, start, end):
    """Fetch one month at a time to keep each request small and retryable."""
    frames = []
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        where = f"datetime >= date'{m0:%Y-%m-%d}' AND datetime < date'{m1:%Y-%m-%d}'"
        params = {"where": where}
        if select:
            params["select"] = select
        url = API.format(ds=ds) + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            r = requests.get(url, timeout=300)
            if r.ok:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
        print(f"  {ds} {m0:%Y-%m}: {len(df):,} rows", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = out[(out["datetime"] >= pd.Timestamp(start, tz="UTC")) &
              (out["datetime"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
    return out.sort_values("datetime").reset_index(drop=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (ds, select) in JOBS.items():
        path = RAW / f"{name}.parquet"
        if path.exists():
            print(f"{path.name} exists, skipping")
            continue
        print(f"Downloading {name} ({ds})")
        df = download(ds, select, START, END)
        df.to_parquet(path, index=False)
        print(f"Saved {path} ({len(df):,} rows, columns: {list(df.columns)})")


if __name__ == "__main__":
    main()
