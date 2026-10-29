"""Compare every saved forecast on the test year and test significance against Elia.

Usage: python scripts/08_compare.py [solar|wind]
Output: results/metrics/<tech>_compare.csv and a printed summary table.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RESULTS  # noqa: E402
from pgtsfm.metrics import daily_loss, dm_test, summarise  # noqa: E402
from pgtsfm.windows import load  # noqa: E402


def main(tech):
    fdir = RESULTS / "forecasts" / tech
    # Leave-one-province-out runs (…_lo-<province>…) cover one province only; they are analysed separately.
    fcs = {p.stem: pd.read_parquet(p) for p in sorted(fdir.glob("*.parquet")) if "_lo-" not in p.stem}
    # Evaluate every model on the same (series, timestamp) rows.
    keys = None
    for fc in fcs.values():
        k = set(zip(fc["series"], fc["ds"]))
        keys = k if keys is None else keys & k
    fcs = {n: fc[[(s, d) in keys for s, d in zip(fc["series"], fc["ds"])]] for n, fc in fcs.items()}

    daylight = None
    if tech == "solar":
        df = load(tech)
        geo = df[df["series"] == df["series"].iloc[0]]
        daylight = pd.DatetimeIndex(geo.loc[geo["cos_zenith"] > 0, "ds"])
    table = summarise(fcs, daylight)
    table["subset"] = "daylight" if daylight is not None else "all hours"

    ref = daily_loss(fcs["elia_da11"])
    rows = []
    for name, fc in fcs.items():
        if name == "elia_da11":
            continue
        stat, p = dm_test(daily_loss(fc), ref)
        rows.append({"model": name, "DM_vs_elia": stat, "p_vs_elia": p})
    table = table.merge(pd.DataFrame(rows), on="model", how="left")
    out = RESULTS / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{tech}_compare.csv", index=False)
    cols = [c for c in ["model", "nMAE", "nRMSE", "bias", "CRPS", "QS3", "PICP80", "PINAW80", "DM_vs_elia", "p_vs_elia"]
            if c in table]
    print(f"== {tech} ({table['subset'].iloc[0]}), test year, all series pooled")
    print(table[table["series"] == "ALL"][cols].sort_values("nMAE").round(3).to_string(index=False))


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["solar", "wind"]):
        main(t)
