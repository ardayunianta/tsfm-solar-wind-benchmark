"""Choose the Chronos-2 settings for the test year from the validation half-year.

For each technology and Chronos-2 variant (power history only, + weather, + weather and Elia), the
setting (7 or 28 days of history; ECMWF only or all three weather models) with the lowest validation
CRPS is chosen ("best"). A second "+ weather" run with 7 days of history (best weather set at 7 days,
"short") is added for the limited-data comparison, where the other models see only 7 days of history.

Reads results/metrics/diagnostics/chronos2_selection.csv (from 13_check_collapse.py --metrics) and
prints one line per test run: <tech> <backbone> <multi 0|1> <context days>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RESULTS  # noqa: E402


def choices():
    """One row per chosen test run: tech, backbone, multi, ctx, role ("best" or "short"), model file name."""
    t = pd.read_csv(RESULTS / "metrics" / "diagnostics" / "chronos2_selection.csv")
    t["backbone"] = t["model"].str.extract(r"zs_(chronos-2(?:-cov(?:-elia)?)?)")[0]
    t["multi"] = t["model"].str.contains("_multi").astype(int)
    t["ctx"] = t["model"].str.extract(r"_ctx(\d+)d")[0].fillna("7").astype(int)
    best = t.loc[t.groupby(["tech", "backbone"])["CRPS"].idxmin()].assign(role="best")
    short = t[(t["backbone"] == "chronos-2-cov") & (t["ctx"] == 7)]
    short = short.loc[short.groupby("tech")["CRPS"].idxmin()].assign(role="short")
    return pd.concat([best, short])[["tech", "backbone", "multi", "ctx", "role", "model"]]


if __name__ == "__main__":
    for r in choices().drop_duplicates(["tech", "model"]).itertuples():
        print(r.tech, r.backbone, r.multi, r.ctx)
