"""Turn the 15-minute tables into day-ahead forecasting samples.

For target day D (00:00-23:45 UTC) the forecast is issued at 09:00 UTC on D-1:
- context  = the 7 days of observed load factor before the issue time (672 steps)
- gap      = 09:00 on D-1 to 00:00 on D (60 steps, not observed at issue time)
- target   = the 96 steps of day D
Weather forecasts and solar geometry for day D are known in advance ("future covariates").
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONTEXT_DAYS, ISSUE_HOUR_UTC, PROCESSED, SPLITS, STEPS_PER_DAY

GAP_STEPS = (24 - ISSUE_HOUR_UTC) * 4
CONTEXT_STEPS = CONTEXT_DAYS * STEPS_PER_DAY


def load(tech):
    df = pd.read_parquet(PROCESSED / f"{tech}.parquet")
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    return df


def split_days(split, first_valid=None):
    a, b = SPLITS[split]
    days = pd.date_range(a, b, freq="D", tz="UTC")
    if first_valid is not None:
        days = days[days >= first_valid]
    return days


@dataclass
class Samples:
    series: str
    days: pd.DatetimeIndex
    context: np.ndarray      # (n_days, CONTEXT_STEPS) load factor before issue time
    target: np.ndarray       # (n_days, 96) true load factor on day D (may contain NaN)
    future: dict             # name -> (n_days, 96) covariates on day D
    capacity: np.ndarray     # (n_days, 96) MW, for converting errors back to MW


def make_samples(series_df, days, future_cols):
    """series_df: rows of one series, indexed continuously at 15 min."""
    s = series_df.set_index("ds").sort_index()
    t0 = s.index[0]
    y = s["y"].to_numpy(dtype=float)
    pos = lambda ts: int((ts - t0) / pd.Timedelta(minutes=15))  # noqa: E731
    keep, ctx, tgt, cap = [], [], [], []
    fut = {c: [] for c in future_cols}
    for d in days:
        issue = d - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC)
        i_issue, i_d = pos(issue), pos(d)
        if i_issue - CONTEXT_STEPS < 0 or i_d + STEPS_PER_DAY > len(y):
            continue
        keep.append(d)
        ctx.append(y[i_issue - CONTEXT_STEPS:i_issue])
        tgt.append(y[i_d:i_d + STEPS_PER_DAY])
        cap.append(s["capacity"].to_numpy(dtype=float)[i_d:i_d + STEPS_PER_DAY])
        for c in future_cols:
            fut[c].append(s[c].to_numpy(dtype=float)[i_d:i_d + STEPS_PER_DAY])
    return Samples(series=str(series_df["series"].iloc[0]), days=pd.DatetimeIndex(keep),
                   context=np.array(ctx), target=np.array(tgt),
                   future={c: np.array(v) for c, v in fut.items()}, capacity=np.array(cap))


def to_frame(samples, point=None, quantiles=None, qlevels=None):
    """Long table of forecasts for one series: series, ds, y, point, q05..q95."""
    n, h = samples.target.shape
    ds = np.concatenate([pd.date_range(d, periods=h, freq="15min") for d in samples.days])
    out = pd.DataFrame({"series": samples.series, "ds": ds, "y": samples.target.reshape(-1),
                        "capacity": samples.capacity.reshape(-1)})
    if point is not None:
        out["point"] = np.asarray(point).reshape(-1)
    if quantiles is not None:
        q = np.asarray(quantiles).reshape(n * h, -1)
        for j, lvl in enumerate(qlevels):
            out[f"q{int(round(lvl * 100)):02d}"] = q[:, j]
        if point is None:
            mid = [j for j, lvl in enumerate(qlevels) if abs(lvl - 0.5) < 1e-9][0]
            out["point"] = q[:, mid]
    return out
