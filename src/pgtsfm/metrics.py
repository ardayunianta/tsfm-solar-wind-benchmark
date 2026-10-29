"""Forecast evaluation. All errors are in load-factor units (share of capacity) and reported in %."""
import numpy as np
import pandas as pd
from scipy import stats


def qcols(df):
    return sorted([c for c in df.columns if c.startswith("q") and c[1:].isdigit()], key=lambda c: int(c[1:]))


def pinball(y, q, level):
    d = y - q
    return np.maximum(level * d, (level - 1) * d)


def evaluate(fc, daylight=None):
    """Metrics for one model's forecast table (rows with missing truth are dropped).

    CRPS is approximated as 2 x the mean pinball loss over the available quantiles; for
    point-only forecasts CRPS equals the MAE.
    """
    df = fc.dropna(subset=["y", "point"])
    if daylight is not None:
        df = df[df["ds"].isin(daylight)]
    e = df["point"] - df["y"]
    out = {"n": len(df), "nMAE": 100 * e.abs().mean(), "nRMSE": 100 * np.sqrt((e ** 2).mean()),
           "bias": 100 * e.mean()}
    qs = qcols(df)
    if qs:
        levels = [int(c[1:]) / 100 for c in qs]
        pl = np.mean([pinball(df["y"].to_numpy(), df[c].to_numpy(), l).mean() for c, l in zip(qs, levels)])
        out["CRPS"] = 100 * 2 * pl
        if {"q10", "q50", "q90"} <= set(qs):
            # Same 3-quantile score for every model, so Elia (which only publishes P10/P50/P90) is comparable.
            out["QS3"] = 100 * 2 * np.mean([pinball(df["y"].to_numpy(), df[c].to_numpy(), l).mean()
                                            for c, l in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9))])
        if "q10" in qs and "q90" in qs:
            inside = (df["y"] >= df["q10"]) & (df["y"] <= df["q90"])
            out["PICP80"] = 100 * inside.mean()
            out["PINAW80"] = 100 * (df["q90"] - df["q10"]).mean()
    else:
        out["CRPS"] = out["nMAE"]
    return out


def daily_loss(fc, kind="abs"):
    df = fc.dropna(subset=["y", "point"]).copy()
    df["day"] = df["ds"].dt.floor("D")
    err = (df["point"] - df["y"]).abs() if kind == "abs" else (df["point"] - df["y"]) ** 2
    return err.groupby([df["series"], df["day"]]).mean()


def dm_test(loss_a, loss_b, h=1):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    Negative statistic = model A has lower loss. Two-sided p-value.
    """
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    var = gamma0 / n
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


def summarise(forecasts, daylight=None):
    """forecasts: {model_name: forecast table}. Returns per-series and overall metric tables."""
    rows = []
    for name, fc in forecasts.items():
        for series, g in fc.groupby("series"):
            rows.append({"model": name, "series": series, **evaluate(g, daylight)})
        rows.append({"model": name, "series": "ALL", **evaluate(fc, daylight)})
    return pd.DataFrame(rows)


def dm_days(fc_a, fc_b, kind="abs"):
    """Diebold-Mariano test on one loss per day: the daily loss of every series, averaged over the
    series both forecasts cover. Regions share the weather, so their errors on a day are correlated;
    pooling (series, day) pairs as independent would overstate significance."""
    la, lb = daily_loss(fc_a, kind), daily_loss(fc_b, kind)
    common = la.index.intersection(lb.index)
    return dm_test(la.loc[common].groupby(level="day").mean(), lb.loc[common].groupby(level="day").mean())


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj
