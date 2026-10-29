"""Economic value of forecasts (E6): imbalance cost with real Belgian prices.

A producer sells its day-ahead forecast f (MWh) at the day-ahead price p_da and settles
the deviation (y - f) at the imbalance price p_imb. Compared with a perfect forecast, it loses
    cost = (f - y) * (p_imb - p_da)      [EUR per quarter-hour]
Summed over the test year this is the imbalance cost; lower is better. Energy per
quarter-hour = load factor x capacity (MW) x 0.25 h.

Day-ahead prices: Energy-Charts API (Fraunhofer ISE; data from Bundesnetzagentur/SMARD,
CC BY 4.0). Imbalance prices: Elia Open Data ods134.
Usage: python scripts/11_value.py [model ...]   (default: all forecasts)
Output: results/metrics/value.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RAW, RESULTS, SPLITS  # noqa: E402


def da_prices():
    path = RAW / "da_price_be.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for m0 in pd.date_range(SPLITS["test"][0], SPLITS["test"][1], freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        for attempt in range(8):  # the API rate-limits bursts of requests
            r = requests.get("https://api.energy-charts.info/price",
                             params={"bzn": "BE", "start": f"{m0:%Y-%m-%d}", "end": f"{m1:%Y-%m-%d}"}, timeout=120)
            if r.status_code != 429:
                break
            time.sleep(30 * (attempt + 1))
        r.raise_for_status()
        time.sleep(5)
        j = r.json()
        frames.append(pd.DataFrame({"ds": pd.to_datetime(j["unix_seconds"], unit="s", utc=True), "p_da": j["price"]}))
    df = pd.concat(frames).drop_duplicates("ds").set_index("ds").sort_index()
    # Hourly prices (before the 15-minute market) are spread over their four quarter-hours.
    df = df.reindex(pd.date_range(df.index.min(), df.index.max() + pd.Timedelta(minutes=45), freq="15min")).ffill()
    df = df.rename_axis("ds").reset_index()
    df.to_parquet(path, index=False)
    return df


def main(models):
    da = da_prices().set_index("ds")["p_da"]
    imb = pd.read_parquet(RAW / "elia_imbalance.parquet")
    imb["datetime"] = pd.to_datetime(imb["datetime"], utc=True)
    imb = imb[imb["resolutioncode"] == "PT15M"].set_index("datetime")["imbalanceprice"]
    prices = pd.DataFrame({"p_da": da, "p_imb": imb}).dropna()
    rows = []
    for tech in ("solar", "wind"):
        for p in sorted((RESULTS / "forecasts" / tech).glob("*.parquet")):
            if models and p.stem not in models:
                continue
            fc = pd.read_parquet(p).dropna(subset=["y", "point"]).join(prices, on="ds", how="inner")
            mwh = 0.25 * fc["capacity"]
            err = (fc["point"] - fc["y"]) * mwh
            spread = fc["p_imb"] - fc["p_da"]
            cost = (err * spread).sum()
            # Sign-blind version: every deviation pays the absolute spread. It removes the effect of
            # errors being correlated with the system imbalance (which can reward a poor forecast).
            cost_abs = (err.abs() * spread.abs()).sum()
            energy = (fc["y"] * mwh).sum()
            rows.append({"tech": tech, "model": p.stem, "imbalance_cost_MEUR": cost / 1e6,
                         "cost_EUR_per_MWh": cost / energy, "abs_spread_cost_MEUR": cost_abs / 1e6,
                         "abs_error_GWh": err.abs().sum() / 1e3,
                         "quarter_hours": len(fc)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "metrics" / "value.csv", index=False)
    for tech, g in out.groupby("tech"):
        print(f"== {tech}")
        print(g.drop(columns="tech").sort_values("imbalance_cost_MEUR").round(3).to_string(index=False))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
