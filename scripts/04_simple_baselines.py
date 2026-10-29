"""Simple baselines and the Elia operational benchmark on the test year.

- persistence:   the latest observed value at the same time of day (D-1 before 09:00, else D-2)
- climatology7: mean of the latest 7 observed days at the same time of day
- elia_da11:     Elia's day-ahead forecast issued at 11:00 local on D-1, with P10/P90
Output: results/forecasts/<tech>/<model>.parquet and results/metrics/<tech>_simple.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import ISSUE_HOUR_UTC, RESULTS, STEPS_PER_DAY  # noqa: E402
from pgtsfm.metrics import summarise  # noqa: E402
from pgtsfm.windows import load, make_samples, split_days, to_frame  # noqa: E402

ISSUE_STEP = ISSUE_HOUR_UTC * 4


def latest_days(context, n_days):
    """(n, n_days, 96): for each time of day on D, the k-th most recent observed value.

    The context ends just before 09:00 on D-1, so times before 09:00 were last seen on D-1
    and later times on D-2. Context index of D-j at time-of-day t is
    (CONTEXT - ISSUE_STEP) - 96 * (j - 1) + t.
    """
    start_d1 = context.shape[1] - ISSUE_STEP
    t = np.arange(STEPS_PER_DAY)
    out = []
    for k in range(n_days):
        j = np.where(t < ISSUE_STEP, k + 1, k + 2)
        out.append(context[:, start_d1 - STEPS_PER_DAY * (j - 1) + t])
    return np.stack(out, axis=1)


def main():
    for tech in ("solar", "wind"):
        df = load(tech)
        days = split_days("test")
        fcs = {"persistence": [], "climatology7": [], "elia_da11": []}
        for series, g in df.groupby("series"):
            smp = make_samples(g, days, ["elia_da11", "elia_da11_p10", "elia_da11_p90"])
            recent = latest_days(smp.context, 7)
            fcs["persistence"].append(to_frame(smp, point=recent[:, 0]))
            fcs["climatology7"].append(to_frame(smp, point=np.nanmean(recent, axis=1)))
            q = np.stack([smp.future["elia_da11_p10"], smp.future["elia_da11"], smp.future["elia_da11_p90"]], -1)
            fcs["elia_da11"].append(to_frame(smp, point=smp.future["elia_da11"], quantiles=q, qlevels=[0.1, 0.5, 0.9]))
        out_dir = RESULTS / "forecasts" / tech
        out_dir.mkdir(parents=True, exist_ok=True)
        fcs = {k: pd.concat(v, ignore_index=True) for k, v in fcs.items()}
        for k, v in fcs.items():
            v.to_parquet(out_dir / f"{k}.parquet", index=False)
        daylight = None
        if tech == "solar":
            geo = df[df["series"] == df["series"].iloc[0]][["ds", "cos_zenith"]]
            daylight = pd.DatetimeIndex(geo.loc[geo["cos_zenith"] > 0, "ds"])
        table = summarise(fcs)
        table["subset"] = "all hours"
        if daylight is not None:
            t2 = summarise(fcs, daylight)
            t2["subset"] = "daylight"
            table = pd.concat([table, t2])
        (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS / "metrics" / f"{tech}_simple.csv", index=False)
        print(f"\n== {tech}")
        print(table[table["series"] == "ALL"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
