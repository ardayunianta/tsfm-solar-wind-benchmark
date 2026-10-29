"""Trained deep-learning baselines (neuralforecast): N-HiTS, TiDE, TFT, LSTM.

Each model sees the last 7 days of load factor and the weather forecasts for the next
156 steps (09:00 UTC on D-1 to 23:45 on D) as future covariates, and outputs quantiles.
Models are trained on everything before the test year (the last 6 months are used for
early stopping) and then produce one forecast per test day from 08:45 UTC cutoffs.

--stride sets the distance between training (and validation) windows. 96 gives one window
per day ending at the 08:45 UTC issue time, the same samples the PG-Adapter and LightGBM
are trained on. The default, 8, gives 12 windows per day (every 2 h, the 08:45 one
included); it was chosen on the validation half-year. Memory grows as 1/stride, because
neuralforecast copies every overlapping window (672 + 156 steps x 27 channels) at each
training step: stride 1 needs about 34 GB for solar, which this 16 GB machine cannot hold.

--scaler sets how inputs are normalised:
  global    each weather input is standardised once with its training-period mean and std;
            the target is already a capacity factor (0-1) and is left as is (default)
  robust    neuralforecast's per-window median/MAD scaling of every input, from the 7-day
            history of that window. For solar, about half of each week is night, so the
            MAD often falls on a dawn/dusk value close to zero and the scaled inputs explode.
  standard  neuralforecast's per-window mean/std scaling

--select is for choosing settings without touching the test year: the test year is removed,
the model is trained on the training split (its last 60 days for early stopping) and it
forecasts the validation half-year. Results go to results/forecasts/<tech>/selection/.

--input-days sets how many days of history each window holds (default 7, like the other models).

Usage: python scripts/09_neuralforecast.py <tech> <model> [--multi] [--stride N] [--scaler S]
                                           [--input-days D] [--select]
Output: results/forecasts/<tech>/nf_<model>[_multi][_strideN][_S][_inDd].parquet
        (each suffix only when the option differs from its default)
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgtsfm.config import ISSUE_HOUR_UTC, QUANTILES, RESULTS, SPLITS  # noqa: E402
from pgtsfm.features import extra_cols  # noqa: E402
from pgtsfm.windows import CONTEXT_STEPS, GAP_STEPS, load  # noqa: E402

H = GAP_STEPS + 96
BASE_EXOG = ["ghi_d2", "dir_d2", "dif_d2", "cc_d2", "t2m_d2", "ws10_d2", "ws100_d2", "wd100_sin_d2", "wd100_cos_d2",
             "cos_zenith", "ghi_clear"]


def make_model(name, exog, max_steps, seed, accelerator, stride, scaler, input_days):
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import LSTM, NHITS, TFT, TiDE
    common = dict(h=H, input_size=input_days * 96, futr_exog_list=exog, loss=MQLoss(quantiles=QUANTILES),
                  max_steps=max_steps, early_stop_patience_steps=5, val_check_steps=100, batch_size=32,
                  step_size=stride, scaler_type="identity" if scaler == "global" else scaler, random_seed=seed,
                  accelerator=accelerator, enable_progress_bar=False,
                  logger=False, enable_model_summary=False)
    if name == "nhits":
        return NHITS(inference_windows_batch_size=512, **common)
    if name == "tide":
        return TiDE(inference_windows_batch_size=512, **common)
    if name == "tft":
        # TFT keeps a 64-unit activation for each of its 26 inputs at every one of the 828 steps
        # (plus 828 x 828 attention scores). With 64 windows per training step this needed ~10 GB
        # and stalled in swap on this 16 GB machine, so it trains on 16 windows per step. Windows
        # are also built from one series per step (batch_size=1) instead of all 11 at once
        # (~4 GB of window copies for solar). Both are memory limits, not tuned settings.
        common["batch_size"] = 1
        return TFT(hidden_size=64, windows_batch_size=16, inference_windows_batch_size=64, **common)
    if name == "lstm":
        return LSTM(encoder_hidden_size=128, decoder_hidden_size=128, inference_windows_batch_size=512, **common)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tech", choices=["solar", "wind"])
    ap.add_argument("model", choices=["nhits", "tide", "tft", "lstm"])
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--accelerator", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=8, help="steps between training windows (96 = one per day)")
    ap.add_argument("--scaler", choices=["global", "robust", "standard"], default="global")
    ap.add_argument("--select", action="store_true", help="hide the test year and forecast the validation days")
    ap.add_argument("--input-days", type=int, default=CONTEXT_STEPS // 96, help="days of history the model sees")
    args = ap.parse_args()
    from neuralforecast import NeuralForecast

    df = load(args.tech)
    exog = BASE_EXOG + (extra_cols("d2") if args.multi else [])
    # Training windows are cut from the start of the data: starting at 09:00 UTC makes every
    # 672-step input end at 08:45 UTC, the issue time (validation and test windows are cut
    # from the end of the data and are aligned to 08:45 by their sizes).
    start = pd.Timestamp(SPLITS["train"][0], tz="UTC") - pd.Timedelta(days=8) + pd.Timedelta(hours=ISSUE_HOUR_UTC)
    nf_df = df[df["ds"] >= start][["series", "ds", "y"] + exog].rename(columns={"series": "unique_id"})
    nf_df["ds"] = nf_df["ds"].dt.tz_localize(None)
    # neuralforecast needs complete data: short gaps are interpolated for training only;
    # evaluation always uses the true (unfilled) values.
    nf_df[["y"] + exog] = nf_df.groupby("unique_id")[["y"] + exog].transform(
        lambda s: s.interpolate(limit_direction="both"))
    if args.scaler == "global":
        # Statistics come from the training split only, so no validation or test information is used.
        train = nf_df["ds"] < pd.Timestamp(SPLITS["train"][1]) + pd.Timedelta(days=1)
        mu, sd = nf_df.loc[train, exog].mean(), nf_df.loc[train, exog].std().replace(0, 1)
        nf_df[exog] = (nf_df[exog] - mu) / sd

    if args.select:
        nf_df = nf_df[nf_df["ds"] < pd.Timestamp(SPLITS["val"][1]) + pd.Timedelta(days=1)]
        eval_days, val_size = pd.date_range(*SPLITS["val"], freq="D"), 60 * 96
    else:
        eval_days, val_size = pd.date_range(*SPLITS["test"], freq="D"), 183 * 96
    n_windows = len(eval_days)
    def fit_predict(accelerator):
        model = make_model(args.model, exog, args.max_steps, args.seed, accelerator,
                           args.stride, args.scaler, args.input_days)
        nf = NeuralForecast(models=[model], freq="15min")
        return model, nf.cross_validation(nf_df, n_windows=n_windows, step_size=96, val_size=val_size, refit=False)

    t0 = time.time()
    try:
        model, cv = fit_predict(args.accelerator)
    except Exception as e:  # some ops are not implemented on Apple MPS
        if args.accelerator == "cpu":
            raise
        print(f"{args.accelerator} failed ({type(e).__name__}: {str(e)[:120]}), retrying on CPU", flush=True)
        model, cv = fit_predict("cpu")
    secs = time.time() - t0

    # Keep the day-D part of each forecast (steps 61..156 after the 08:45 cutoff).
    cv["day"] = cv["ds"].dt.floor("D")
    cv = cv[cv["day"] == (cv["cutoff"] + pd.Timedelta(minutes=15)).dt.floor("D") + pd.Timedelta(days=1)]
    mcol = model.__class__.__name__ if hasattr(model, "__class__") else args.model
    # MQLoss names quantile columns by interval level: "-lo-90.0" = q0.05, "-median", "-hi-90.0" = q0.95.
    qmap = {}
    for c in cv.columns:
        m = re.match(rf"{mcol}-(lo|hi)-([0-9.]+)$", c)
        if m:
            lvl = float(m.group(2))
            qmap[c] = round((100 - lvl) / 200 if m.group(1) == "lo" else (100 + lvl) / 200, 4)
        elif c == f"{mcol}-median":
            qmap[c] = 0.5
    if not qmap:
        raise RuntimeError(f"Could not find quantile columns in {list(cv.columns)}")
    qs = sorted(qmap.items(), key=lambda kv: kv[1])
    out = pd.DataFrame({"series": cv["unique_id"].to_numpy(), "ds": cv["ds"].dt.tz_localize("UTC").to_numpy()})
    q = np.clip(np.sort(cv[[c for c, _ in qs]].to_numpy(), axis=1), 0, 1)
    for j, (_, lvl) in enumerate(qs):
        out[f"q{int(round(lvl * 100)):02d}"] = q[:, j]
    if args.tech == "solar":
        night = df.set_index(["series", "ds"])["cos_zenith"].reindex(pd.MultiIndex.from_frame(out[["series", "ds"]])) <= 0
        out.loc[night.to_numpy(), [f"q{int(round(l * 100)):02d}" for _, l in qs]] = 0.0
    truth = df.set_index(["series", "ds"])[["y", "capacity"]]
    out = out.join(truth, on=["series", "ds"])
    out["point"] = out["q50"]
    name = (f"nf_{args.model}{'_multi' if args.multi else ''}{f'_stride{args.stride}' if args.stride != 8 else ''}"
            f"{f'_{args.scaler}' if args.scaler != 'global' else ''}"
            f"{f'_in{args.input_days}d' if args.input_days != CONTEXT_STEPS // 96 else ''}")
    folder = RESULTS / "forecasts" / args.tech / ("selection" if args.select else "")
    folder.mkdir(parents=True, exist_ok=True)
    out.to_parquet(folder / f"{name}.parquet", index=False)
    print(f"Saved {folder.name}/{name}.parquet: {len(out):,} rows, {secs:.0f} s")
    if args.select:
        from pgtsfm.metrics import summarise
        geo = df[df["series"] == df["series"].iloc[0]]  # same daylight definition as 08_compare.py
        daylight = pd.DatetimeIndex(geo.loc[geo["cos_zenith"] > 0, "ds"]) if args.tech == "solar" else None
        row = summarise({name: out}, daylight).query("series == 'ALL'").iloc[0]
        print(f"validation: nMAE {row['nMAE']:.3f}  CRPS {row['CRPS']:.3f}  PICP80 {row['PICP80']:.1f}")


if __name__ == "__main__":
    main()
