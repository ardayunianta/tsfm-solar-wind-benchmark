"""Per-step features for day D: weather forecasts, physics proxies, calendar and recent power."""
import numpy as np
import pandas as pd

from .config import ISSUE_HOUR_UTC, STEPS_PER_DAY

NWP_BASE = ["ghi", "dir", "dif", "cc", "t2m", "ws10", "ws100", "wd100_sin", "wd100_cos"]
GEO = ["cos_zenith", "ghi_clear"]
# Extra weather models (ICON-EU, GFS) used in the multi-model setting.
EXTRA_PREFIXES = ["icon_", "gfs_"]
EXTRA_BASE = ["ghi", "cc", "t2m", "ws10", "ws100", "wd100_sin", "wd100_cos"]


def extra_cols(lag="d2"):
    return [f"{p}{c}_{lag}" for p in EXTRA_PREFIXES for c in EXTRA_BASE]


def future_cols(lag="d2", multi=False):
    return [f"{c}_{lag}" for c in NWP_BASE] + GEO + (extra_cols(lag) if multi else [])


def ensemble_mean(future, var, lag, multi):
    """Mean of a weather variable across the available models (ECMWF only unless multi)."""
    arrs = [future[f"{var}_{lag}"]] + ([future[f"{p}{var}_{lag}"] for p in EXTRA_PREFIXES] if multi else [])
    return np.nanmean(np.stack(arrs), axis=0)


def recent_power(context):
    """Summary of observed power at issue time, broadcast to the 96 target steps."""
    issue = ISSUE_HOUR_UTC * 4
    start_d1 = context.shape[1] - issue
    t = np.arange(STEPS_PER_DAY)
    same_tod = []
    for k in range(7):
        j = np.where(t < issue, k + 1, k + 2)
        same_tod.append(context[:, start_d1 - STEPS_PER_DAY * (j - 1) + t])
    same_tod = np.stack(same_tod, axis=1)
    n = context.shape[0]
    rep = lambda v: np.repeat(v[:, None], STEPS_PER_DAY, axis=1)  # noqa: E731
    return {
        "p_persist": same_tod[:, 0],
        "p_clim7": np.nanmean(same_tod, axis=1),
        "p_last": rep(context[:, -1]),
        "p_mean24h": rep(np.nanmean(context[:, -STEPS_PER_DAY:], axis=1)),
        "p_max7d": rep(np.nanmax(context, axis=1)),
    }


def step_table(samples, lag="d2", multi=False):
    """Flatten Samples into one row per (day, step) with all features."""
    n, h = samples.target.shape
    cols = {}
    for c in NWP_BASE:
        cols[c] = samples.future[f"{c}_{lag}"]
    if multi:
        for c in extra_cols(lag):
            cols[c] = samples.future[c]
    for c in GEO:
        cols[c] = samples.future[c]
    ghi_clear = samples.future["ghi_clear"]
    ghi = ensemble_mean(samples.future, "ghi", lag, multi)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["csi"] = np.where(ghi_clear > 10, ghi / ghi_clear, np.nan).clip(0, 1.5)
    cols["ws100_cubed"] = np.clip(ensemble_mean(samples.future, "ws100", lag, multi) / 3.6, 0, 30) ** 3
    cols.update(recent_power(samples.context))
    doy = samples.days.dayofyear.to_numpy()
    cols["tod"] = np.tile(np.arange(h), (n, 1))
    cols["doy_sin"] = np.repeat(np.sin(2 * np.pi * doy / 365.25)[:, None], h, axis=1)
    cols["doy_cos"] = np.repeat(np.cos(2 * np.pi * doy / 365.25)[:, None], h, axis=1)
    out = pd.DataFrame({k: np.asarray(v, dtype=float).reshape(-1) for k, v in cols.items()})
    out["series"] = samples.series
    out["y"] = samples.target.reshape(-1)
    return out
