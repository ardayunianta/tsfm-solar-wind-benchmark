"""Online adaptive conformal calibration of the 80 % interval (E5).

For each series and test day D, the interval [q10, q90] is widened (or narrowed) by an
amount learned only from days that were fully observed at issue time (up to D-2):
  * conformity score per step: s = max(q10 - y, y - q90)   (negative when y is inside)
  * the correction is the (1 - alpha_t) empirical quantile of the scores of the last 60 days
  * alpha_t follows adaptive conformal inference (Gibbs & Candes, 2021):
        alpha_{t+1} = alpha_t + gamma * (0.2 - miss_rate_t)
For solar, only daylight steps are used and night stays at zero.

Usage: python scripts/12_conformal.py <tech> <model> [<model> ...]
Output: results/forecasts/<tech>/<model>_aci.parquet and results/metrics/<tech>_coverage_by_season.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RESULTS  # noqa: E402

TARGET_MISS, GAMMA, WINDOW = 0.2, 0.02, 60


def calibrate(fc):
    fc = fc.sort_values(["series", "ds"]).copy()
    fc["day"] = fc["ds"].dt.floor("D")
    active = fc["q90"] > 0  # daylight for solar (night forecasts are exactly 0); always true for wind
    out = []
    for _, g in fc.groupby("series"):
        g = g.copy()
        days = g["day"].unique()
        score = np.maximum(g["q10"] - g["y"], g["y"] - g["q90"])
        alpha, hist = TARGET_MISS, []
        lo, hi = g["q10"].to_numpy().copy(), g["q90"].to_numpy().copy()
        for i, d in enumerate(days):
            m = (g["day"] == d).to_numpy()
            act = active[g.index].to_numpy() & m
            past = [s for s in hist[max(0, i - 1 - WINDOW):max(0, i - 1)]]  # days up to D-2
            if past:
                pool = np.concatenate(past)
                level = min(max(1 - alpha, 0.0), 1.0)
                corr = np.quantile(pool, level) if len(pool) else 0.0
                lo[act] -= corr
                hi[act] += corr
            # Update alpha with this day's realised miss rate (used from D+2 on, see `past`).
            y = g["y"].to_numpy()[act]
            ok = ~np.isnan(y)
            if ok.any():
                miss = np.mean((y[ok] < lo[act][ok]) | (y[ok] > hi[act][ok]))
                alpha = alpha + GAMMA * (TARGET_MISS - miss)
            sc = score.to_numpy()[act]
            hist.append(sc[~np.isnan(sc)])
        g["q10"], g["q90"] = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        out.append(g)
    res = pd.concat(out).drop(columns="day")
    qs = sorted([c for c in res.columns if c.startswith("q") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    res[qs] = np.sort(res[qs].to_numpy(), axis=1)  # keep quantiles ordered after the adjustment
    return res


def season(ts):
    return ts.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})


def coverage_table(fcs):
    rows = []
    for name, fc in fcs.items():
        f = fc.dropna(subset=["y"])
        f = f[f["q90"] > 0]
        f = f.assign(season=season(f["ds"]), inside=(f["y"] >= f["q10"]) & (f["y"] <= f["q90"]),
                     width=f["q90"] - f["q10"])
        for s, g in f.groupby("season"):
            rows.append({"model": name, "season": s, "PICP80": 100 * g["inside"].mean(), "PINAW80": 100 * g["width"].mean()})
    return pd.DataFrame(rows)


def main(tech, models):
    d = RESULTS / "forecasts" / tech
    fcs = {}
    for m in models:
        fc = pd.read_parquet(d / f"{m}.parquet")
        cal = calibrate(fc)
        cal.to_parquet(d / f"{m}_aci.parquet", index=False)
        fcs[m], fcs[f"{m}_aci"] = fc, cal
    t = coverage_table(fcs)
    t.to_csv(RESULTS / "metrics" / f"{tech}_coverage_by_season.csv", index=False)
    print(t.pivot(index="model", columns="season", values="PICP80").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
