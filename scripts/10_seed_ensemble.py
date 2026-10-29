"""Average PG-Adapter forecasts over random seeds (quantile averaging).

Usage: python scripts/10_seed_ensemble.py <tech> <tag-without-seed>
  e.g. python scripts/10_seed_ensemble.py wind chronos-bolt-base_multi_roll30d_d2
Reads results/forecasts/<tech>/pga_<tag>_s*.parquet, writes pga_<tag>_ens.parquet.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import RESULTS  # noqa: E402
from pgtsfm.metrics import qcols  # noqa: E402


def main(tech, tag):
    d = RESULTS / "forecasts" / tech
    # Only plain seed runs (…_s0, …_s1, …), not derived files such as …_s0_aci.
    files = sorted(f for f in d.glob(f"pga_{tag}_s*.parquet") if re.fullmatch(rf"pga_{re.escape(tag)}_s\d+", f.stem))
    if not files:
        raise SystemExit(f"no files for {tag}")
    frames = [pd.read_parquet(f) for f in files]
    out = frames[0].copy()
    qs = qcols(out)
    out[qs] = np.sort(np.mean([f[qs].to_numpy() for f in frames], axis=0), axis=1)
    out["point"] = out["q50"]
    out.to_parquet(d / f"pga_{tag}_ens.parquet", index=False)
    print(f"{tech}/pga_{tag}_ens from {len(files)} seeds")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
