"""Choose the test-year settings of TimesFM 2.5, Moirai 2.0, TiRex and fine-tuned Chronos-2 on validation.

Same rule as choose_chronos2.py: for each technology and model variant, the validation run with the
lowest CRPS on the validation half-year is used on the test year.

  zero-shot (16_other_tsfms.py): per model (tirex, timesfm-2.5, timesfm-2.5-cov, moirai-2.0),
      7 or 28 days of history and, with covariates, ECMWF only or all three weather models.
      Scores: results/metrics/diagnostics/other_tsfm_selection.csv
  fine-tuned (15_finetune_chronos2.py): per variant (cov, cov-elia), history and weather set,
      learning rate. Scores: results/metrics/diagnostics/finetune_selection.csv

Both score files come from 13_check_collapse.py --metrics. Prints one line per test run:
  zs <tech> <model> <multi 0|1> <context days>
  ft <tech> <variant> <multi 0|1> <context days> <mode> <learning rate> <seed>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RESULTS  # noqa: E402

DIAG = RESULTS / "metrics" / "diagnostics"
ZS = r"zs_(?P<backbone>[a-z0-9.-]+?)(?P<multi>_multi)?(?:_ctx(?P<ctx>\d+)d)?$"
FT = (r"ft_chronos-2-(?P<variant>cov(?:-elia)?)(?P<multi>_multi)?_ctx(?P<ctx>\d+)d_(?P<mode>full|lora)"
      r"_lr(?P<lr>[0-9.e-]+)_s(?P<seed>\d+)$")


def best(t, pattern, by):
    """Parse the run names and keep the lowest-CRPS run per technology and `by`."""
    t = pd.concat([t, t["model"].str.extract(pattern)], axis=1).dropna(subset=[by])
    t["multi"] = t["multi"].notna().astype(int)
    t["ctx"] = t["ctx"].fillna("7").astype(int)
    return t.loc[t.groupby(["tech", by])["CRPS"].idxmin()].reset_index(drop=True)


def zs_choices():
    return best(pd.read_csv(DIAG / "other_tsfm_selection.csv"), ZS, "backbone")


def ft_choices():
    return best(pd.read_csv(DIAG / "finetune_selection.csv"), FT, "variant")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "zs"
    if kind == "zs":
        for r in zs_choices().itertuples():
            print("zs", r.tech, r.backbone, r.multi, r.ctx)
    else:
        for r in ft_choices().itertuples():
            print("ft", r.tech, r.variant, r.multi, r.ctx, r.mode, r.lr, r.seed)
