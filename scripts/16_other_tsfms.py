"""Zero-shot forecasts from TimesFM 2.5, TimesFM 3.0, Moirai 2.0 and TiRex, built like the Chronos-2 ones.

These models need packages whose versions conflict with the main environment (Moirai pins
torch 2.4 and numpy 1.26), so each runs in its own conda environment and the work is split in
three steps. Only "predict" runs in a model environment; it needs nothing but numpy and the model.

  export   (main env)   the inputs of every day-ahead sample, built with windows() from
                        06_tsfm_priors.py: power history before 09:00 UTC on D-1, and the
                        covariates over that history and over the 156-step horizon
                        -> results/cache/windows/<tech>/<split>_ctx<N>d_<covs>.npz
  predict  (model env)  quantile forecasts -> results/cache/quantiles/<tech>/<split>/<name>.npz
  collect  (main env)   standard forecast tables -> results/forecasts/<tech>/[selection/]<name>.parquet

Covariates (the same weather forecasts and solar geometry as Chronos-2 gets):
  TimesFM 2.5  in-context linear regression on the covariates ("xreg + timesfm", the package
               default): a regression fitted on the history, TimesFM forecasts its residuals
  TimesFM 3.0  covariates as extra input series, read by the model itself (attention across series).
               Only run with covariates, as an extra row: it was released in August 2026, during the
               test year (see the paper)
  Moirai 2.0   power history only. Its paper (Liu et al. 2025, arXiv 2511.11698) drops covariate
               support; the package still accepts them as extra variates. moirai-2.0-cov does that,
               and is kept only for the check reported in the paper (wind, validation, all three
               weather models): worse than without covariates, and hours instead of under a minute.
               Scores: results/metrics/diagnostics/moirai_covariates_check.csv
  TiRex        power history only (no covariate input)
All four models output the quantiles 0.1, ..., 0.9; as for Chronos-Bolt, the 0.05 and 0.95
levels are set to the 0.1 and 0.9 forecasts.

Usage:
  python scripts/16_other_tsfms.py export  --tech solar --days val --context-days 7 [--multi] [--no-covs]
  <model env>/python scripts/16_other_tsfms.py predict timesfm-2.5-cov --tech solar --days val --context-days 7 [--multi]
  python scripts/16_other_tsfms.py collect timesfm-2.5-cov --tech solar --days val --context-days 7 [--multi]
Model names: tirex, timesfm-2.5, timesfm-2.5-cov, timesfm-3.0-cov, moirai-2.0, moirai-2.0-cov.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Same folder as pgtsfm.config.RESULTS, which the model environments cannot import
CACHE = Path(os.environ.get("PGTSFM_RESULTS", ROOT / "results")) / "cache"
HORIZON = 156  # 60 steps from 09:00 UTC on D-1 to midnight, then the 96 steps of day D
MODELS = ["tirex", "timesfm-2.5", "timesfm-2.5-cov", "timesfm-3.0-cov", "moirai-2.0", "moirai-2.0-cov"]
QLEVELS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  # = pgtsfm.config.QUANTILES


def window_file(a, covs):
    tag = "multi" if covs and a.multi else ("ecmwf" if covs else "none")
    return CACHE / "windows" / a.tech / f"{a.days}_ctx{a.context_days}d_{tag}.npz"


def run_name(a):
    return f"zs_{a.model}{'_multi' if a.multi else ''}{f'_ctx{a.context_days}d' if a.context_days != 7 else ''}"


def fill_missing(context):
    """Missing power values: linear interpolation between observed ones (constant at the ends)."""
    x = np.arange(context.shape[1])
    return np.array([np.interp(x, np.flatnonzero(~np.isnan(c)), c[~np.isnan(c)]) if (~np.isnan(c)).any()
                     else np.zeros_like(c) for c in context])


def nine_to_eleven(q9):
    """(..., 9) forecasts at 0.1..0.9 -> (..., 11) at QLEVELS, with 0.05 = 0.1 and 0.95 = 0.9."""
    return np.concatenate([q9[..., :1], q9, q9[..., -1:]], axis=-1)


# ---------------------------------------------------------------------------------------------
# export / collect (main environment)

def tsfm_module():
    spec = importlib.util.spec_from_file_location("tsfm", ROOT / "scripts" / "06_tsfm_priors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def export(a):
    sys.path.insert(0, str(ROOT / "src"))
    from pgtsfm.windows import load, make_samples, split_days
    tsfm = tsfm_module()
    covs = [] if a.no_covs else tsfm.covariate_names("chronos-2-cov", a.multi)
    arrays = {}
    for series, g in load(a.tech).groupby("series"):
        smp = make_samples(g, split_days(a.days), [])
        ctx, past, fut = tsfm.windows(g, smp.days, covs, a.context_days * 96)
        arrays[f"{series}|days"] = smp.days.as_unit("ns").asi8
        arrays[f"{series}|context"] = ctx.astype(np.float32)
        if covs:
            arrays[f"{series}|covs"] = np.concatenate([past, fut], axis=1).astype(np.float32)
    out = window_file(a, covs)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cov_names=np.array(covs), **arrays)
    print(f"Saved {out.relative_to(CACHE.parents[1])}")


def collect(a):
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd
    from pgtsfm.config import QUANTILES, RESULTS
    from pgtsfm.windows import load, make_samples, split_days, to_frame
    assert list(QUANTILES) == QLEVELS
    name = run_name(a)
    q_all = np.load(CACHE / "quantiles" / a.tech / a.days / f"{name}.npz")
    frames = []
    for series, g in load(a.tech).groupby("series"):
        smp = make_samples(g, split_days(a.days), ["cos_zenith"])
        assert np.array_equal(q_all[f"{series}|days"], smp.days.as_unit("ns").asi8)
        q = np.clip(q_all[f"{series}|q"], 0, 1)
        q = np.sort(q, axis=-1)  # no crossing quantiles
        if a.tech == "solar":
            q[smp.future["cos_zenith"] <= 0] = 0.0
        frames.append(to_frame(smp, quantiles=q, qlevels=QUANTILES))
    fdir = RESULTS / "forecasts" / a.tech / ("selection" if a.days == "val" else "")
    fdir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(fdir / f"{name}.parquet", index=False)
    print(f"Saved {(fdir / f'{name}.parquet').relative_to(RESULTS / 'forecasts')}")


# ---------------------------------------------------------------------------------------------
# predict (model environments). Each function maps context (n, L) and covariates (n, L + 156, F)
# or None to quantiles (n, 156, 11).

def predict_tirex(context, covs, batch=256):
    import torch
    from tirex import load_model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model("NX-AI/TiRex", backend="torch", device=device)
    q, _ = model.forecast(context=torch.tensor(context), prediction_length=HORIZON, output_type="numpy",
                          batch_size=batch)
    return nine_to_eleven(np.asarray(q)[:, :HORIZON])


def predict_timesfm(context, covs, batch=64):
    import timesfm
    # torch.compile only speeds up long runs; on this CPU it took longer than the forecasts themselves
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    model.compile(timesfm.ForecastConfig(
        max_context=context.shape[1], max_horizon=HORIZON, per_core_batch_size=batch, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
        fix_quantile_crossing=True, return_backcast=covs is not None))
    context = fill_missing(context)  # the covariate regression cannot fit through missing values
    out = []
    for i in range(0, len(context), batch):
        inputs = list(context[i:i + batch].astype(np.float64))
        if covs is None:
            _, q = model.forecast(horizon=HORIZON, inputs=inputs)
        else:
            c = covs[i:i + batch].astype(np.float64)
            _, q = model.forecast_with_covariates(
                inputs=inputs, dynamic_numerical_covariates={f"c{k}": list(c[:, :, k]) for k in range(c.shape[2])},
                xreg_mode="xreg + timesfm", force_on_cpu=True)
        out.append(np.stack([np.asarray(x)[:HORIZON] for x in q]))
    q10 = np.concatenate(out)  # last axis: mean, then the quantiles 0.1, ..., 0.9
    return nine_to_eleven(q10[..., 1:])


def predict_timesfm3(context, covs, batch=16):
    import torch
    from timesfm3 import TimesFM3Evaluator
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # The evaluator class sets the package's benchmark defaults: averaging over the series and its
    # negative, no negative forecasts for non-negative series, sorted quantiles
    model = TimesFM3Evaluator.from_pretrained("google/timesfm-3.0-pytorch", device=device, per_core_batch_size=batch,
                                              revision="43046b85ec22d584a13f8098c2ed39c889e129c2")  # 2 Sept 2026
    pf = None if covs is None else list(covs.transpose(0, 2, 1))  # (n, F, L + 156): one row per covariate
    out = model.predict_batch(contexts=list(context), horizon=HORIZON, past_future_covariates=pf)
    return nine_to_eleven(np.stack([o.quantiles for o in out]))  # (n, 156, 9) at 0.1, ..., 0.9


def predict_moirai(context, covs, batch=32):
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    device = "cpu"  # small model; one of its operations (cummax) is not implemented on Apple GPUs
    L, F = context.shape[1], (0 if covs is None else covs.shape[2])
    model = Moirai2Forecast(module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                            prediction_length=HORIZON, context_length=L, target_dim=1,
                            feat_dynamic_real_dim=F, past_feat_dynamic_real_dim=0).to(device).eval()

    def tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    out = []
    for i in range(0, len(context), batch):
        ctx = context[i:i + batch]
        observed = ~np.isnan(ctx)  # filled values are marked as unobserved
        kw = dict(past_target=tensor(fill_missing(ctx)[..., None]), past_observed_target=torch.tensor(observed[..., None],
                  device=device), past_is_pad=torch.zeros(ctx.shape, dtype=torch.bool, device=device))
        if F:
            c = covs[i:i + batch]
            kw.update(feat_dynamic_real=tensor(c), observed_feat_dynamic_real=torch.ones(c.shape, dtype=torch.bool,
                                                                                           device=device))
        with torch.no_grad():
            q = model(**kw)  # (n, 9, 156)
        out.append(q.float().cpu().numpy().transpose(0, 2, 1))
    return nine_to_eleven(np.concatenate(out))


def predict(a):
    use_covs = a.model.endswith("-cov")
    w = np.load(window_file(a, ["cov"] if use_covs else []))
    series = sorted({k.split("|")[0] for k in w.files if "|" in k})
    context = np.concatenate([w[f"{s}|context"] for s in series])
    covs = np.concatenate([w[f"{s}|covs"] for s in series]) if use_covs else None
    t0 = time.time()
    fn = {"tirex": predict_tirex, "timesfm-2.5": predict_timesfm, "timesfm-3.0": predict_timesfm3,
          "moirai-2.0": predict_moirai}[a.model.removesuffix("-cov")]
    q = fn(context, covs)
    assert q.shape == (len(context), HORIZON, len(QLEVELS)), q.shape
    q = q[:, -96:].astype(np.float32)
    out, start = {}, 0
    for s in series:
        n = len(w[f"{s}|days"])
        out[f"{s}|days"], out[f"{s}|q"] = w[f"{s}|days"], q[start:start + n]
        start += n
    path = CACHE / "quantiles" / a.tech / a.days / f"{run_name(a)}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"{a.model} {a.tech} {a.days}: {len(context)} forecasts in {time.time() - t0:.0f} s -> "
          f"{path.relative_to(CACHE.parents[1])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "predict", "collect"])
    ap.add_argument("model", nargs="?", choices=MODELS)
    ap.add_argument("--tech", choices=["solar", "wind"], required=True)
    ap.add_argument("--days", choices=["val", "test"], default="val")
    ap.add_argument("--context-days", type=int, default=7)
    ap.add_argument("--multi", action="store_true", help="covariates also from ICON-EU and GFS")
    ap.add_argument("--no-covs", action="store_true", help="export: power history only")
    a = ap.parse_args()
    if a.step != "export" and a.model is None:
        ap.error("predict and collect need a model name")
    if a.multi and a.model and not a.model.endswith("-cov"):
        ap.error("--multi only applies to the -cov models")
    {"export": export, "predict": predict, "collect": collect}[a.step](a)


if __name__ == "__main__":
    main()
