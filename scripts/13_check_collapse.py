"""Find forecasts that collapsed to zero.

A series-month counts as collapsed when the median forecast is exactly 0 for more than 95 % of
the quarter-hours with real output (measured load factor above 2 %). A trained model that does
this has stopped learning: with the training loss computed on quantiles clipped to [0, 1], a
median pushed below 0 gets no gradient (see PGAdapter in src/pgtsfm/adapter.py).

--metrics also adds nMAE, CRPS and 80 % coverage over all series (solar: daylight hours, as in
08_compare.py), e.g. to compare training options on the --dev months in selection/.

Usage: python scripts/13_check_collapse.py [glob] [--folder selection] [--metrics] [--out NAME]
       e.g. python scripts/13_check_collapse.py "pga_*roll*" --folder selection --metrics
Output: results/metrics/diagnostics/<NAME>.csv (default collapse[_<folder>]), one row per forecast file
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RESULTS  # noqa: E402
from pgtsfm.metrics import summarise  # noqa: E402
from pgtsfm.windows import load  # noqa: E402


def collapsed_months(fc):
    """Series-months whose median forecast is 0 for > 95 % of the steps with measured output."""
    live = fc[fc["y"] > 0.02]
    zero = (live["point"] == 0).groupby([live["series"], live["ds"].dt.strftime("%Y-%m")]).mean()
    return len(zero), zero[zero > 0.95].index.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob", nargs="?", default="pga_*")
    ap.add_argument("--folder", default="", help="subfolder of results/forecasts/<tech>/, e.g. selection")
    ap.add_argument("--metrics", action="store_true", help="also compute nMAE, CRPS and PICP80")
    ap.add_argument("--out", help="output file name without .csv (default: collapse[_<folder>])")
    args = ap.parse_args()
    rows = []
    for tech in ("solar", "wind"):
        paths = sorted((RESULTS / "forecasts" / tech / args.folder).glob(f"{args.glob}.parquet"))
        daylight = None
        if args.metrics and tech == "solar" and paths:
            df = load(tech)
            geo = df[df["series"] == df["series"].iloc[0]]  # same daylight definition as 08_compare.py
            daylight = pd.DatetimeIndex(geo.loc[geo["cos_zenith"] > 0, "ds"])
        for p in paths:
            fc = pd.read_parquet(p)
            n, dead = collapsed_months(fc)
            row = dict(tech=tech, model=p.stem, series_months=n, collapsed=len(dead),
                       months=" ".join(sorted({m for _, m in dead})))
            if args.metrics:
                m = summarise({p.stem: fc}, daylight).query("series == 'ALL'").iloc[0]
                row.update(nMAE=m["nMAE"], CRPS=m["CRPS"], PICP80=m["PICP80"])
            rows.append(row)
    if not rows:
        sys.exit(f"no forecast files match {args.glob} in {args.folder or 'the main folders'}")
    t = pd.DataFrame(rows)
    out = RESULTS / "metrics" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / f"{args.out or 'collapse' + ('_' + args.folder if args.folder else '')}.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"\n{t['collapsed'].sum()} of {t['series_months'].sum()} series-months collapsed in {len(t)} files")


if __name__ == "__main__":
    main()
