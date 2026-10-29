"""Train and evaluate PG-Adapter.

Examples:
  python scripts/07_train_adapter.py --tech solar --backbone chronos-bolt-base
  python scripts/07_train_adapter.py --tech wind --backbone chronos-bolt-base --train-days 30
  python scripts/07_train_adapter.py --tech solar --backbone none            # ablation: no foundation-model prior
  python scripts/07_train_adapter.py --tech solar --backbone chronos-bolt-base --no-physics
  python scripts/07_train_adapter.py --tech solar --backbone chronos-bolt-base --multi --train-days 7 --rolling

Training data: the last N days of the training period (all 365 by default), for all series
of one technology. Early stopping uses the validation half-year in the full-data setting,
and the last 20 % of the N days in the limited-data setting (a new plant has no separate
validation year). The model is then refit on all available days for the chosen number of
epochs and evaluated on the test year.

--rolling (limited-data experiment E2): a new model is trained at the start of every test
month on the N days that end two days before the month (the last days whose power is fully
observed at the 09:00 UTC issue time on the eve of the month) and forecasts that month. This
emulates a plant with only N days of power history (each training day also has its 7-day
context), without a gap of months between training and use. Weather, solar-geometry and
calendar inputs are standardised with the 365 days before the month, because weather-forecast
archives exist for any site; the recent-power inputs are capacity factors (0-1) and are left
unscaled, since scaling them would need more than N days of power data.

--dev (with --rolling): the same monthly retraining, but for the 12 months before the test year
(2024-10 to 2025-09), to compare training options without touching the test year. The scaling
window is cut to the days available since 2024-04-01. Results go to results/forecasts/<tech>/selection/.

--clipped-loss: compute the training loss on the quantiles after clipping them to [0, 1], as in the
first version (results/forecasts/<tech>/superseded/clipped_loss/). Clipped quantiles get no gradient,
and some limited-data models collapsed to an all-zero forecast, so the default now trains on the
unclipped quantiles (see PGAdapter). Forecasts are clipped either way.
Output: results/forecasts/<tech>/pga_<tag>.parquet and results/runs/<tech>/pga_<tag>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.adapter import PGAdapter, pinball_loss  # noqa: E402
from pgtsfm.config import QUANTILES, RESULTS, SPLITS  # noqa: E402
from pgtsfm.features import NWP_BASE, ensemble_mean, extra_cols, future_cols, recent_power  # noqa: E402
from pgtsfm.windows import load, make_samples, split_days, to_frame  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ELIA_COLS = ["elia_da11_p10", "elia_da11", "elia_da11_p90"]
RECENT = ("p_persist", "p_clim7", "p_last", "p_mean24h", "p_max7d")
# Position of the recent-power columns in "cov" (only the two calendar columns come after them).
POWER_COLS = slice(-2 - len(RECENT), -2)
# --dev: the 12 months before the test year (each needs at least 90 earlier days from 2024-04-01).
DEV_MONTHS = ("2024-10-01", SPLITS["val"][1])


def build_arrays(samples, prior, lag, multi=False):
    """Stack per-step covariates (n, 96, F), raw physics inputs and night mask."""
    f = samples.future
    ghi_clear = f["ghi_clear"]
    with np.errstate(divide="ignore", invalid="ignore"):
        csi = np.where(ghi_clear > 10, ensemble_mean(f, "ghi", lag, multi) / ghi_clear, 0.0).clip(0, 1.5)
    rp = recent_power(samples.context)
    n = samples.target.shape[0]
    doy = samples.days.dayofyear.to_numpy()
    feats = [f[f"{c}_{lag}"] for c in NWP_BASE] + ([f[c] for c in extra_cols(lag)] if multi else []) + \
            [f["cos_zenith"], ghi_clear / 1000.0, csi] + \
            [rp[k] for k in RECENT] + \
            [np.repeat(np.sin(2 * np.pi * doy / 365.25)[:, None], 96, 1),
             np.repeat(np.cos(2 * np.pi * doy / 365.25)[:, None], 96, 1)]
    cov = np.nan_to_num(np.stack(feats, -1).astype(np.float32))
    raw = {"ghi": ensemble_mean(f, "ghi", lag, multi), "ghi_clear": ghi_clear,
           "t2m": ensemble_mean(f, "t2m", lag, multi), "ws100": ensemble_mean(f, "ws100", lag, multi)}
    raw = {k: np.nan_to_num(v).astype(np.float32) for k, v in raw.items()}
    night = f["cos_zenith"] <= 0
    pri = prior if prior is not None else np.zeros((n, 96, 1), np.float32)
    return {"cov": cov, "raw": raw, "night": night, "prior": np.nan_to_num(pri).astype(np.float32),
            "y": samples.target.astype(np.float32)}


def concat(parts):
    out = {}
    for k in parts[0]:
        if k == "raw":
            out[k] = {r: np.concatenate([p[k][r] for p in parts]) for r in parts[0][k]}
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


def take(a, idx):
    return {k: ({r: v[idx] for r, v in a[k].items()} if k == "raw" else a[k][idx]) for k in a}


def to_t(a, tech):
    t = {"cov": torch.tensor(a["cov"]), "prior": torch.tensor(a["prior"]), "y": torch.tensor(a["y"]),
         "raw": {k: torch.tensor(v) for k, v in a["raw"].items()},
         "night": torch.tensor(a["night"]) if tech == "solar" else None}
    return t


def batches(t, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield (t["prior"][j].to(DEV), t["cov"][j].to(DEV), {k: v[j].to(DEV) for k, v in t["raw"].items()},
               None if t["night"] is None else t["night"][j].to(DEV), t["y"][j].to(DEV))


def run_epochs(model, tr, va, epochs, levels, lr=1e-3, patience=20, bs=64, seed=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_ep, best_state, history = np.inf, 0, None, []
    n = len(tr["y"])
    for ep in range(1, epochs + 1):
        model.train()
        for p, c, r, m, y in batches(tr, rng.permutation(n), bs):
            loss = pinball_loss(model(p, c, r, m), y, levels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if va is None:
            continue
        vl = evaluate_loss(model, va, levels)
        history.append(vl)
        if vl < best - 1e-6:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ep, best, history


@torch.no_grad()
def evaluate_loss(model, t, levels):
    model.eval()
    tot, cnt = 0.0, 0
    for p, c, r, m, y in batches(t, np.arange(len(t["y"])), 256):
        k = (~torch.isnan(y)).sum().item()
        tot += pinball_loss(model(p, c, r, m), y, levels).item() * k
        cnt += k
    return tot / max(cnt, 1)


@torch.no_grad()
def predict(model, t):
    model.eval()
    out = [model(p, c, r, m).cpu().numpy() for p, c, r, m, _ in batches(t, np.arange(len(t["y"])), 256)]
    return np.concatenate(out)


def load_prior(tech, backbone):
    if backbone == "none":
        return None
    z = np.load(RESULTS / "priors" / tech / f"{backbone}.npz")
    out = {}
    for key in z.files:
        s, kind = key.split("|")
        out.setdefault(s, {})[kind] = z[key]
    def to_days(x):
        # Older cache files store microseconds (pandas 3 default), newer ones nanoseconds.
        return pd.to_datetime(x, unit="us" if x.max() < 1e17 else "ns", utc=True)
    return {s: pd.Series(list(v["q"]), index=to_days(v["days"])) for s, v in out.items()}


def series_arrays(df, days, args, priors, use_elia):
    """(series, samples, model arrays) for every series of one technology over the given days."""
    out = []
    for series, g in df.groupby("series"):
        smp = make_samples(g, days, future_cols(args.lag, args.multi) + (ELIA_COLS if use_elia else []))
        pri = []
        if priors is not None:
            pri.append(np.stack(priors[series].reindex(smp.days).to_list()))
        if use_elia:
            pri.append(np.stack([smp.future[c] for c in ELIA_COLS], -1))
        pri = np.concatenate(pri, -1) if pri else None
        out.append((series, smp, build_arrays(smp, pri, args.lag, args.multi)))
    return out


def standardise(arrays, mu, sd):
    for a in arrays:
        a["cov"] = (a["cov"] - mu) / sd


def early_stop_split(a, n_series):
    """Limited data: the last 20 % of each series' days (at most 30) are held out for early stopping.
    The arrays must hold the same number of consecutive days for every series, series after series."""
    per = len(a["y"]) // n_series
    cut = max(int(per * 0.8), per - 30)
    idx_fit = np.concatenate([np.arange(i * per, i * per + cut) for i in range(n_series)])
    idx_stop = np.concatenate([np.arange(i * per + cut, (i + 1) * per) for i in range(n_series)])
    return take(a, idx_fit), take(a, idx_stop)


def model_factory(a, args, use_elia):
    n_prior = a["prior"].shape[-1]
    prior_mid = n_prior - 2 if use_elia else None  # Elia's median when the operator forecast is a prior
    return lambda: PGAdapter(n_prior=n_prior, n_cov=a["cov"].shape[-1], quantiles=QUANTILES, tech=args.tech,
                             use_prior=args.backbone != "none", use_physics=not args.no_physics,
                             prior_mid=prior_mid, clip_train=args.clipped_loss).to(DEV)


def fit_model(new_model, fit, stop, refit, epochs, levels, seed, tech):
    """Choose the number of epochs by early stopping on `stop`, then retrain from scratch on `refit`."""
    torch.manual_seed(seed)
    model = new_model()
    best_ep, best_val, _ = run_epochs(model, to_t(fit, tech), to_t(stop, tech), epochs, levels, seed=seed)
    torch.manual_seed(seed)
    model = new_model()
    run_epochs(model, to_t(refit, tech), None, max(best_ep, 5), levels, seed=seed)
    return model, best_ep, best_val


def fixed_split(args, df, priors, use_elia, levels):
    """One model trained before the test year (full data, limited data or leave-one-out) and used all year."""
    sets = {"train": split_days("train")[-args.train_days:], "val": split_days("val"), "test": split_days("test")}
    per_split = {k: series_arrays(df, days, args, priors, use_elia) for k, days in sets.items()}
    if args.holdout:  # E3: train/validate on the other series, test only on the held-out one
        per_split = {k: [x for x in v if (x[0] == args.holdout) == (k == "test")] for k, v in per_split.items()}
    arrays = {k: concat([a for _, _, a in v]) for k, v in per_split.items()}
    test_samples = [smp for _, smp, _ in per_split["test"]]

    # Standardise covariates with training statistics only.
    flat = arrays["train"]["cov"].reshape(-1, arrays["train"]["cov"].shape[-1])
    standardise(arrays.values(), flat.mean(0), flat.std(0) + 1e-6)
    if args.train_days >= 365:
        fit, stop = arrays["train"], arrays["val"]
        refit = concat([arrays["train"], arrays["val"]])  # everything available before the test year
    else:
        fit, stop = early_stop_split(arrays["train"], len(per_split["train"]))
        refit = arrays["train"]

    t0 = time.time()
    model, best_ep, best_val = fit_model(model_factory(arrays["train"], args, use_elia), fit, stop, refit,
                                         args.epochs, levels, args.seed, args.tech)
    train_s = time.time() - t0
    t1 = time.time()
    q = predict(model, to_t(arrays["test"], args.tech))
    infer_s = time.time() - t1
    frames, i = [], 0
    for smp in test_samples:
        k = len(smp.days)
        frames.append(to_frame(smp, quantiles=q[i:i + k], qlevels=QUANTILES))
        i += k
    info = dict(best_epoch=best_ep, best_val_pinball=best_val, train_seconds=train_s, infer_seconds=infer_s)
    return model, frames, info


def rolling_months(args, df, priors, use_elia, levels):
    """E2: a new model at the start of every test month, trained on the N days just before it."""
    every = series_arrays(df, pd.date_range(SPLITS["train"][0], SPLITS["test"][1], freq="D", tz="UTC"),
                          args, priors, use_elia)
    n_series = len(every)

    def pick(days):  # rows for these days, series after series (days before the data start are skipped)
        return concat([take(a, np.flatnonzero(smp.days.isin(days))) for _, smp, a in every])

    test_days = pd.date_range(*DEV_MONTHS, freq="D", tz="UTC") if args.dev else split_days("test")
    preds, months, train_s, infer_s = [[] for _ in every], [], 0.0, 0.0
    for m0 in test_days[test_days.day == 1]:
        last = m0 - pd.Timedelta(days=2)  # fully observed at the 09:00 UTC issue time on the eve of m0
        month_days = test_days[(test_days >= m0) & (test_days < m0 + pd.offsets.MonthBegin(1))]
        tr, scale, te = (pick(pd.date_range(end=last, periods=args.train_days, freq="D")),
                         pick(pd.date_range(end=last, periods=365, freq="D")), pick(month_days))
        assert len(tr["y"]) == n_series * args.train_days and len(te["y"]) == n_series * len(month_days)
        flat = scale["cov"].reshape(-1, scale["cov"].shape[-1])
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        mu[POWER_COLS], sd[POWER_COLS] = 0.0, 1.0  # recent-power inputs stay capacity factors
        standardise([tr, te], mu, sd)
        fit, stop = early_stop_split(tr, n_series)

        t0 = time.time()
        model, best_ep, best_val = fit_model(model_factory(tr, args, use_elia), fit, stop, tr,
                                             args.epochs, levels, args.seed, args.tech)
        t1 = time.time()
        q = predict(model, to_t(te, args.tech)).reshape(n_series, len(month_days), *te["y"].shape[1:], -1)
        train_s, infer_s = train_s + t1 - t0, infer_s + time.time() - t1
        for i in range(n_series):
            preds[i].append(q[i])
        months.append(dict(month=m0.strftime("%Y-%m"), best_epoch=best_ep, best_val_pinball=best_val))
        print(f"  {months[-1]['month']}: best epoch {best_ep}, early-stopping pinball {best_val:.4f}", flush=True)

    frames = []
    for (_, g), p in zip(df.groupby("series"), preds):
        smp = make_samples(g, test_days, [])
        frames.append(to_frame(smp, quantiles=np.concatenate(p), qlevels=QUANTILES))
    info = dict(months=months, train_seconds=train_s, infer_seconds=infer_s)
    return model, frames, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True, choices=["solar", "wind"])
    ap.add_argument("--backbone", default="chronos-bolt-base")
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--lag", default="d2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--multi", action="store_true", help="add ICON-EU and GFS weather forecasts")
    ap.add_argument("--holdout", default=None, help="series left out of training and used alone for testing (E3)")
    ap.add_argument("--rolling", action="store_true", help="E2: retrain every test month on the N days before it")
    ap.add_argument("--dev", action="store_true", help="with --rolling: forecast the 12 months before the test year")
    ap.add_argument("--clipped-loss", action="store_true", help="first version: train on quantiles clipped to [0, 1]")
    args = ap.parse_args()
    if args.rolling and (args.train_days >= 365 or args.holdout):
        ap.error("--rolling is the limited-data experiment: use it with --train-days below 365 and no --holdout")
    if args.dev and not args.rolling:
        ap.error("--dev only works with --rolling")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load(args.tech)
    # --backbone: a foundation model, "elia" (operator forecast as prior; Setting B), both joined
    # with "+", or "none". Setting B is issued at 10:00 UTC, when Elia's 11:00-local forecast is public.
    parts = args.backbone.split("+")
    use_elia = "elia" in parts
    fm = [p for p in parts if p not in ("elia", "none")]
    priors = load_prior(args.tech, fm[0]) if fm else None
    levels = torch.tensor(QUANTILES, device=DEV)
    run = rolling_months if args.rolling else fixed_split
    model, frames, fit_info = run(args, df, priors, use_elia, levels)

    # The seed goes last so that 10_seed_ensemble.py can average runs that differ only by seed.
    tag = (f"{args.backbone}{'_nophys' if args.no_physics else ''}{'_multi' if args.multi else ''}"
           f"_{'roll' if args.rolling else ''}{args.train_days}d_{args.lag}"
           + (f"_lo-{args.holdout}" if args.holdout else "") + ("_clipped" if args.clipped_loss else "")
           + f"_s{args.seed}")
    sub = "selection" if args.dev else ""
    out = RESULTS / "forecasts" / args.tech / sub
    out.mkdir(parents=True, exist_ok=True)
    fc = pd.concat(frames, ignore_index=True)
    fc.to_parquet(out / f"pga_{tag}.parquet", index=False)
    rdir = RESULTS / "runs" / args.tech / sub
    rdir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    info = dict(vars(args), **fit_info, n_params=n_params, device=DEV)
    (rdir / f"pga_{tag}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    if args.dev:
        from pgtsfm.metrics import summarise
        geo = df[df["series"] == df["series"].iloc[0]]  # same daylight definition as 08_compare.py
        daylight = pd.DatetimeIndex(geo.loc[geo["cos_zenith"] > 0, "ds"]) if args.tech == "solar" else None
        row = summarise({tag: fc}, daylight).query("series == 'ALL'").iloc[0]
        print(f"dev months: nMAE {row['nMAE']:.3f}  CRPS {row['CRPS']:.3f}  PICP80 {row['PICP80']:.1f}")


if __name__ == "__main__":
    main()
