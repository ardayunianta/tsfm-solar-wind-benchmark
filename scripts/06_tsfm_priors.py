"""Zero-shot foundation-model forecasts for every series and day.

The backbone is frozen, so its forecasts are computed once and cached. They serve two
purposes: zero-shot baselines on the test year (RQ1) and the "prior" input of PG-Adapter.

The forecast is issued at 09:00 UTC on D-1, so the model predicts 60 + 96 = 156 steps and
the last 96 (day D) are kept.

Chronos-2 can also read covariates: their values over the power history ("past") and their known
values over the 156-step horizon ("future"). "chronos-2-cov" gets the weather forecasts and solar
geometry that the trained models get (--multi: also the ICON-EU and GFS forecasts);
"chronos-2-cov-elia" also gets Elia's day-ahead forecast (Setting B, like PG-Adapter with
--backbone elia). Missing covariate values are interpolated; missing power values stay missing.

--days all    train, validation and test days: caches the PG-Adapter prior and saves the test forecasts
--days val    the validation half-year only, to choose settings (results/forecasts/<tech>/selection/)
--days test   the test year only
--context-days: days of power history the model sees (default 7, like the other models; the
               PG-Adapter prior always uses 7).

Usage: python scripts/06_tsfm_priors.py <backbone> [--days all|val|test] [--multi] [--context-days N]
                                       [--tech solar|wind]
Output: results/priors/<tech>/<backbone>.npz (--days all) and
        results/forecasts/<tech>/[selection/]zs_<backbone>[_multi][_ctxNd].parquet
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import ISSUE_HOUR_UTC, QUANTILES, RESULTS  # noqa: E402
from pgtsfm.features import future_cols  # noqa: E402
from pgtsfm.windows import CONTEXT_STEPS, GAP_STEPS, load, make_samples, split_days, to_frame  # noqa: E402

BACKBONES = {
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-bolt-base": "amazon/chronos-bolt-base",
    "chronos-2": "amazon/chronos-2",            # power history only
    "chronos-2-cov": "amazon/chronos-2",        # plus weather forecasts and solar geometry
    "chronos-2-cov-elia": "amazon/chronos-2",   # plus Elia's day-ahead forecast
}
HORIZON = GAP_STEPS + 96
ELIA_COLS = ["elia_da11_p10", "elia_da11", "elia_da11_p90"]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_pipeline(name):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(BACKBONES[name], device_map=device(), torch_dtype=torch.float32)


def covariate_names(name, multi):
    if "-cov" not in name:
        return []
    return future_cols("d2", multi) + (ELIA_COLS if name.endswith("-elia") else [])


def windows(series_df, days, covs, context_steps, with_target=False):
    """Power history before the issue time; covariates over that history (past) and the horizon (future).
    with_target=True also returns the observed power over the 156-step horizon (for fine-tuning)."""
    s = series_df.set_index("ds").sort_index()
    t0 = s.index[0]
    y = s["y"].to_numpy(dtype=float)
    cov = s[covs].interpolate(limit_direction="both").to_numpy(dtype=float) if covs else None
    ctx, past, fut, target = [], [], [], []
    for d in days:
        issue = d - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC)
        i = int((issue - t0) / pd.Timedelta(minutes=15))
        if i < context_steps:
            raise ValueError(f"less than {context_steps} steps of history before {d:%Y-%m-%d}")
        ctx.append(y[i - context_steps:i])
        target.append(y[i:i + HORIZON])
        if covs:
            past.append(cov[i - context_steps:i])
            fut.append(cov[i:i + HORIZON])
    out = np.array(ctx), (np.array(past) if covs else None), (np.array(fut) if covs else None)
    return out + (np.array(target),) if with_target else out


def predict(pipe, context, covs=(), past_cov=None, fut_cov=None, batch=128):
    """context (n, L) with NaN for missing -> quantiles (n, 96, Q)."""
    outs = []
    for i in range(0, len(context), batch):
        if not covs:
            ctx = torch.tensor(context[i:i + batch], dtype=torch.float32)
            if type(pipe).__name__ == "Chronos2Pipeline":  # expects (series, variates, time)
                ctx = ctx.unsqueeze(1)
            q, _ = pipe.predict_quantiles(ctx, prediction_length=HORIZON, quantile_levels=QUANTILES)
            if isinstance(q, list):
                q = torch.stack([x.squeeze(0) for x in q])
        else:
            inputs = [{"target": context[j],
                       "past_covariates": {c: past_cov[j, :, k] for k, c in enumerate(covs)},
                       "future_covariates": {c: fut_cov[j, :, k] for k, c in enumerate(covs)}}
                      for j in range(i, min(i + batch, len(context)))]
            q, _ = pipe.predict_quantiles(inputs, prediction_length=HORIZON, quantile_levels=QUANTILES)
            q = torch.stack([x.squeeze(0) for x in q])
        outs.append(q[:, -96:, :].float().cpu().numpy())
    return np.clip(np.concatenate(outs), 0, 1)


def subset(smp, mask):
    return type(smp)(smp.series, smp.days[mask], smp.context[mask], smp.target[mask],
                     {k: v[mask] for k, v in smp.future.items()}, smp.capacity[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", nargs="?", default="chronos-bolt-base", choices=list(BACKBONES))
    ap.add_argument("--days", choices=["all", "val", "test"], default="all")
    ap.add_argument("--multi", action="store_true", help="covariates also from ICON-EU and GFS")
    ap.add_argument("--context-days", type=int, default=CONTEXT_STEPS // 96)
    ap.add_argument("--tech", choices=["solar", "wind"], help="one technology only (default: both)")
    args = ap.parse_args()
    if args.days == "all" and args.context_days * 96 != CONTEXT_STEPS:
        ap.error("the PG-Adapter prior uses the 7-day context: use --days val or test with --context-days")
    if args.multi and "-cov" not in args.backbone:
        ap.error("--multi only applies to the -cov backbones")
    covs = covariate_names(args.backbone, args.multi)
    splits = ("train", "val", "test") if args.days == "all" else (args.days,)
    name = (f"zs_{args.backbone}{'_multi' if args.multi else ''}"
            f"{f'_ctx{args.context_days}d' if args.context_days * 96 != CONTEXT_STEPS else ''}")
    pipe = load_pipeline(args.backbone)
    for tech in (args.tech,) if args.tech else ("solar", "wind"):
        df = load(tech)
        days = pd.DatetimeIndex(np.concatenate([split_days(s) for s in splits]))
        cache, frames = {}, []
        t0 = time.time()
        for series, g in df.groupby("series"):
            smp = make_samples(g, days, ["cos_zenith"])
            context, past, fut = windows(g, smp.days, covs, args.context_days * 96)
            q = predict(pipe, context, covs, past, fut, batch=32 if covs else 128)
            if tech == "solar":
                q[smp.future["cos_zenith"] <= 0] = 0.0
            cache[series] = {"days": smp.days.as_unit("ns").asi8, "q": q.astype(np.float32)}
            keep = smp.days >= split_days("test")[0] if args.days == "all" else np.ones(len(smp.days), bool)
            frames.append(to_frame(subset(smp, keep), quantiles=q[keep], qlevels=QUANTILES))
            print(f"  {tech}/{series}: {len(smp.days)} days", flush=True)
        if args.days == "all":
            out = RESULTS / "priors" / tech
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{args.backbone}.npz",
                                **{f"{s}|{k}": v for s, d in cache.items() for k, v in d.items()})
        fdir = RESULTS / "forecasts" / tech / ("selection" if args.days == "val" else "")
        fdir.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(fdir / f"{name}.parquet", index=False)
        print(f"Saved {tech}/{fdir.name}/{name}.parquet in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
