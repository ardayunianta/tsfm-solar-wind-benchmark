"""LightGBM quantile regression: a strong non-deep baseline with the same weather inputs.

One global model per technology and quantile, trained on all series. The number of
boosting rounds is chosen on the validation half-year, then the model is refit on
train + validation and evaluated on the test year.

days=N trains on the last N training days instead (the last 20 % of them for early stopping).
With "rolling" (limited-data experiment E2) a new model is trained at the start of every test
month on the N days that end two days before it, as in 07_train_adapter.py --rolling.

"elia" adds Elia's day-ahead forecast (median, P10, P90) as inputs, the same information that
PG-Adapter gets with --backbone elia (Setting B: issued once Elia's forecast is public).
Output: results/forecasts/<tech>/lightgbm[_multi][_elia][_Nd | _rollNd].parquet
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import QUANTILES, RESULTS  # noqa: E402
from pgtsfm.features import future_cols, step_table  # noqa: E402
from pgtsfm.windows import load, make_samples, split_days, to_frame  # noqa: E402

ELIA_COLS = ["elia_da11_p10", "elia_da11", "elia_da11_p90"]
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)


def tables(df, days, lag, multi, elia=False):
    out, smps = [], []
    for _, g in df.groupby("series"):
        smp = make_samples(g, days, future_cols(lag, multi) + (ELIA_COLS if elia else []))
        smps.append(smp)
        t = step_table(smp, lag, multi)
        for c in ELIA_COLS if elia else []:
            t[c] = smp.future[c].reshape(-1)
        t["day"] = smp.days.repeat(smp.target.shape[1])  # used to select rows, not a feature
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["series"] = t["series"].astype("category")
    return t, smps


def align_series(*tabs):
    """Give every table the same series categories, so LightGBM codes them identically."""
    cats = sorted(set().union(*(t["series"].cat.categories for t in tabs)))
    for t in tabs:
        t["series"] = pd.Categorical(t["series"].astype(str), categories=cats)


def fit_quantiles(tr, va, te, feats):
    """Rounds per quantile chosen by early stopping on `va`, then refit on `tr` + `va` to predict `te`."""
    full = pd.concat([tr, va])  # refit on all available days
    preds, rounds = [], []
    for q in QUANTILES:
        p = dict(PARAMS, objective="quantile", alpha=q)
        m = lgb.train(p, lgb.Dataset(tr[feats], tr["y"]), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(va[feats], va["y"])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds.append(m.best_iteration)
        m = lgb.train(p, lgb.Dataset(full[feats], full["y"]), num_boost_round=m.best_iteration)
        preds.append(m.predict(te[feats]))
    return np.stack(preds, axis=1), rounds


def main(lag="d2", multi=False, train_days=365, techs=("solar", "wind"), rolling=False, elia=False):
    for tech in techs:
        df = load(tech)
        test_days = split_days("test")
        te, te_smps = tables(df, test_days, lag, multi, elia)
        feats = [c for c in te.columns if c not in ("y", "day")]
        cut = max(int(train_days * 0.8), 1)  # limited data: first 80 % of the days to fit, the rest to stop
        if rolling:
            hist, _ = tables(df, pd.date_range(test_days[0] - pd.Timedelta(days=train_days + 1), test_days[-1],
                                               freq="D"), lag, multi, elia)
            align_series(hist, te)
            hist = hist.dropna(subset=["y"])
            qpred = np.zeros((len(te), len(QUANTILES)))
            for m0 in test_days[test_days.day == 1]:
                # The last training day is fully observed at the 09:00 UTC issue time on the eve of m0.
                days = pd.date_range(end=m0 - pd.Timedelta(days=2), periods=train_days, freq="D")
                month = ((te["day"] >= m0) & (te["day"] < m0 + pd.offsets.MonthBegin(1))).to_numpy()
                qpred[month], rounds = fit_quantiles(hist[hist["day"].isin(days[:cut])],
                                                     hist[hist["day"].isin(days[cut:])], te[month], feats)
                print(f"  {tech} {m0:%Y-%m}: {min(rounds)}-{max(rounds)} rounds", flush=True)
        else:
            if train_days >= 365:
                tr, _ = tables(df, split_days("train"), lag, multi, elia)
                va, _ = tables(df, split_days("val"), lag, multi, elia)
            else:
                # Limited data: the last N training days; the last 20 % of them for early stopping.
                days = split_days("train")[-train_days:]
                tr, _ = tables(df, days[:cut], lag, multi, elia)
                va, _ = tables(df, days[cut:], lag, multi, elia)
            align_series(tr, va, te)
            qpred, rounds = fit_quantiles(tr.dropna(subset=["y"]), va.dropna(subset=["y"]), te, feats)
            for q, r in zip(QUANTILES, rounds):
                print(f"  {tech} q={q:.2f}: {r} rounds", flush=True)
        qpred = np.sort(np.clip(qpred, 0, 1), axis=1)  # fix quantile crossing
        if tech == "solar":
            qpred[te["cos_zenith"].to_numpy() <= 0] = 0.0
        frames, i = [], 0
        for smp in te_smps:
            k = smp.target.size
            frames.append(to_frame(smp, quantiles=qpred[i:i + k].reshape(*smp.target.shape, -1), qlevels=QUANTILES))
            i += k
        out = RESULTS / "forecasts" / tech
        out.mkdir(parents=True, exist_ok=True)
        name = ("lightgbm" + ("" if lag == "d2" else f"_{lag}") + ("_multi" if multi else "") +
                ("_elia" if elia else "") + ("" if train_days >= 365 else f"_{'roll' if rolling else ''}{train_days}d"))
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{name}.parquet", index=False)
        print(f"Saved {tech}/{name}")


if __name__ == "__main__":
    # python scripts/05_lightgbm.py [lag] [multi] [elia] [days=N] [rolling] [tech=solar|wind]
    opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    main(sys.argv[1] if len(sys.argv) > 1 else "d2", "multi" in sys.argv[2:], int(opts.get("days", 365)),
         (opts["tech"],) if "tech" in opts else ("solar", "wind"), "rolling" in sys.argv[2:], "elia" in sys.argv[2:])
