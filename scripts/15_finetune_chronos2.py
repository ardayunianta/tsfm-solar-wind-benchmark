"""Fine-tune Chronos-2 on the training year (the fourth way of using a foundation model).

Each training example is one day-ahead sample built exactly like the zero-shot forecasts in
06_tsfm_priors.py: the power history before 09:00 UTC on D-1, the covariates over that history,
and their values over the 156-step horizon, with the observed power over the horizon as the
target. Every example is passed as its own short series with min_past = the history length, so
the only window Chronos-2 can cut from it is the issue-time window (no random cut points).

Training uses the training year only. The validation half-year is used twice, as for the other
deep-learning models: a sample of its days picks the checkpoint (lowest validation loss, checked
every --eval-every steps), and the full half-year scores each setting (--days val), so that the
setting used on the test year is chosen on validation only (scripts/choose_finetune.py).

Usage:
  python scripts/15_finetune_chronos2.py --tech solar --multi --context-days 28 --mode lora --lr 1e-4
         [--variant cov|cov-elia] [--steps 1000] [--batch-size 64] [--eval-every 100] [--seed 0]
  python scripts/15_finetune_chronos2.py ... --days test   (loads the saved checkpoint, forecasts the test year)
Output: results/models/chronos2_ft/<tech>/<name>/ (checkpoint and training log),
        results/forecasts/<tech>/selection/<name>.parquet (--days val) or <tech>/<name>.parquet (--days test)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pgtsfm.config import QUANTILES, RESULTS  # noqa: E402
from pgtsfm.windows import load, make_samples, split_days, to_frame  # noqa: E402

spec = importlib.util.spec_from_file_location("tsfm", ROOT / "scripts" / "06_tsfm_priors.py")
tsfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsfm)


def run_name(a):
    return (f"ft_chronos-2-{a.variant}{'_multi' if a.multi else ''}_ctx{a.context_days}d"
            f"_{a.mode}_lr{a.lr:g}_s{a.seed}")


def examples(df, days, covs, context_steps):
    """One dict per series and day: target and covariates over history + horizon."""
    out = []
    for _, g in df.groupby("series"):
        smp = make_samples(g, days, [])
        ctx, past, fut, target = tsfm.windows(g, smp.days, covs, context_steps, with_target=True)
        for j in range(len(ctx)):
            out.append({"target": np.concatenate([ctx[j], target[j]]).astype(np.float32),
                        "past_covariates": {c: np.concatenate([past[j, :, k], fut[j, :, k]]).astype(np.float32)
                                            for k, c in enumerate(covs)}})
    return out


def prepared(dicts, covs):
    from chronos.chronos2.preprocess import from_list_of_dicts
    return from_list_of_dicts(dicts, prediction_length=tsfm.HORIZON, known_covariates_names=covs)


def finetune(a, tech, covs, outdir):
    from chronos import BaseChronosPipeline
    df = load(tech)
    L = a.context_days * 96
    # The data start on 15 March 2024, so with 28 days of history the first training days are skipped.
    first = (df["ds"].min() + pd.Timedelta(days=a.context_days + 1)).normalize()
    train = examples(df, split_days("train", first_valid=first), covs, L)
    val_days = split_days("val")[::a.val_stride]  # a sample of validation days picks the checkpoint
    val = examples(df, val_days, covs, L)
    print(f"{tech}: {len(train)} training and {len(val)} checkpoint-selection examples, "
          f"{1 + len(covs)} rows each", flush=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=tsfm.device(),
                                               torch_dtype=torch.float32)
    t0 = time.time()
    ft = pipe.fit(prepared(train, covs), prediction_length=tsfm.HORIZON,
                  validation_inputs=prepared(val, covs), finetune_mode=a.mode, context_length=L,
                  learning_rate=a.lr, num_steps=a.steps, batch_size=a.batch_size, min_past=L,
                  output_dir=outdir, seed=a.seed, eval_steps=a.eval_every, save_steps=a.eval_every,
                  logging_steps=a.eval_every, disable_tqdm=True)
    minutes = (time.time() - t0) / 60
    # fit() saves the best checkpoint to finetuned-ckpt/ (LoRA: the adapter only). Merged weights are
    # saved to model/, so that --days test loads a plain Chronos-2 model either way.
    if a.mode == "lora":
        ft.model = ft.model.merge_and_unload()
    ft.save_pretrained(outdir / "model")
    (outdir / "train_info.json").write_text(json.dumps(
        {"minutes": round(minutes, 1), "train_examples": len(train), "val_examples": len(val),
         "args": vars(a)}, indent=1))
    print(f"Fine-tuned in {minutes:.0f} min -> {outdir / 'model'}", flush=True)
    return ft


def forecast(pipe, tech, covs, days_split, context_steps, name):
    df = load(tech)
    frames = []
    t0 = time.time()
    for series, g in df.groupby("series"):
        smp = make_samples(g, split_days(days_split), ["cos_zenith"])
        context, past, fut = tsfm.windows(g, smp.days, covs, context_steps)
        q = tsfm.predict(pipe, context, covs, past, fut, batch=32)
        if tech == "solar":
            q[smp.future["cos_zenith"] <= 0] = 0.0
        frames.append(to_frame(smp, quantiles=q, qlevels=QUANTILES))
    fdir = RESULTS / "forecasts" / tech / ("selection" if days_split == "val" else "")
    fdir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(fdir / f"{name}.parquet", index=False)
    print(f"Saved {(fdir / f'{name}.parquet').relative_to(RESULTS / 'forecasts')} in {time.time() - t0:.0f} s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", choices=["solar", "wind"], required=True)
    ap.add_argument("--variant", choices=["cov", "cov-elia"], default="cov")
    ap.add_argument("--multi", action="store_true", help="covariates also from ICON-EU and GFS")
    ap.add_argument("--context-days", type=int, default=28)
    ap.add_argument("--mode", choices=["full", "lora"], default="lora")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64, help="rows per batch (target + covariates)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-stride", type=int, default=4, help="every k-th validation day picks the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", choices=["val", "test"], default="val")
    a = ap.parse_args()
    covs = tsfm.covariate_names(f"chronos-2-{a.variant}", a.multi)
    name = run_name(a)
    outdir = RESULTS / "models" / "chronos2_ft" / a.tech / name
    done = RESULTS / "forecasts" / a.tech / ("selection" if a.days == "val" else "") / f"{name}.parquet"
    if done.exists():
        print(f"{done.relative_to(RESULTS)} exists, skipped")
        return
    if a.days == "val":
        pipe = finetune(a, a.tech, covs, outdir)
    else:
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(outdir / "model", device_map=tsfm.device(),
                                                   torch_dtype=torch.float32)
    forecast(pipe, a.tech, covs, a.days, a.context_days * 96, name)


if __name__ == "__main__":
    main()
