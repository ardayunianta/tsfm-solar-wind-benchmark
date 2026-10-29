"""Compute cost: wall-clock time of every model in the main table, measured on one technology.

Each model's own pipeline (the same scripts and settings as its reported forecasts) is run once, one job at a
time, from the processed data to the test-year forecasts, including any training, prior computation and model
loading. All outputs go to a temporary results folder (PGTSFM_RESULTS), so no reported forecast is touched.
Settings are read from the forecasts in the main table (14_make_tables.resolve), so they match the chosen ones.

Counted per model: training (if any) plus forecasting the test year. Not counted: choosing settings on the
validation half-year, and for fine-tuned Chronos-2 the forecast of the validation half-year that
15_finetune_chronos2.py makes after training (its training time is read from train_info.json instead).
Trained models use one seed; PG-Adapter's reported forecasts average three.

Usage: python scripts/17_timing.py [--tech wind] [--only KEY ...] [--redo] [--keep]
Output: results/metrics/timing.csv (one row per step) and results/metrics/timing_machine.json
        (--redo moves the earlier rows of the --only models to results/metrics/timing_superseded.csv)
"""
import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pgtsfm.config import RESULTS  # noqa: E402

ENVS = Path(os.environ.get("PGTSFM_ENVS", "/opt/miniconda3/envs"))
PY = ENVS / "pgtsfm" / "bin" / "python"
MODEL_ENV = {"tirex": "tsfm-tirex", "timesfm-2.5": "tsfm-timesfm", "timesfm-3.0": "tsfm-timesfm",
             "moirai-2.0": "tsfm-moirai"}
S = ROOT / "scripts"
CPU, GPU = "CPU", "GPU (MPS)"


def table_names(tech):
    """Forecast file name of every main-table row, as resolved by the table script."""
    spec = importlib.util.spec_from_file_location("mt", S / "14_make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return {k: mt.resolve(k, tech) for k in mt.ROWS}


def zs_flags(name):
    multi = "_multi" in name
    ctx = re.search(r"_ctx(\d+)d", name)
    return (["--multi"] if multi else []) + ["--context-days", ctx.group(1) if ctx else "7"]


def jobs(tech, names):
    """key -> (device, [(step, command, counted)])."""
    j = {
        "lgb": (CPU, [("train + forecast", [PY, S / "05_lightgbm.py", "d2", "multi", f"tech={tech}"], True)]),
        "lgbElia": (CPU, [("train + forecast",
                           [PY, S / "05_lightgbm.py", "d2", "multi", "elia", f"tech={tech}"], True)]),
        "nhits": (CPU, [("train + forecast", [PY, S / "09_neuralforecast.py", tech, "nhits", "--multi",
                                              "--input-days", "1", "--accelerator", "cpu"], True)]),
        "tide": (CPU, [("train + forecast", [PY, S / "09_neuralforecast.py", tech, "tide", "--multi",
                                             "--accelerator", "cpu"], True)]),
        "tft": (GPU, [("train + forecast", [PY, S / "09_neuralforecast.py", tech, "tft", "--multi"], True)]),
        "lstm": (GPU, [("train + forecast", [PY, S / "09_neuralforecast.py", tech, "lstm", "--multi"], True)]),
        "pgaNone": (GPU, [("train + forecast", [PY, S / "07_train_adapter.py", "--tech", tech, "--backbone",
                                                "none", "--multi", "--seed", "0"], True)]),
        "pgaBolt": (GPU, [("prior on all days", [PY, S / "06_tsfm_priors.py", "chronos-bolt-base", "--days",
                                                 "all", "--tech", tech], True),
                          ("train + forecast", [PY, S / "07_train_adapter.py", "--tech", tech, "--backbone",
                                                "chronos-bolt-base", "--multi", "--seed", "0"], True)]),
        "pgaElia": (GPU, [("train + forecast", [PY, S / "07_train_adapter.py", "--tech", tech, "--backbone",
                                                "elia", "--multi", "--seed", "0"], True)]),
    }
    # Zero-shot Chronos models (06_tsfm_priors.py), with the settings of the reported forecast
    for key, backbone in [("bolt", "chronos-bolt-base"), ("c2", "chronos-2"), ("c2cov", "chronos-2-cov"),
                          ("c2covElia", "chronos-2-cov-elia")]:
        j[key] = (GPU, [("forecast", [PY, S / "06_tsfm_priors.py", backbone, "--days", "test", "--tech", tech]
                         + zs_flags(names[key]), True)])
    # The other zero-shot models (16_other_tsfms.py): export windows, predict in the model's own environment,
    # collect the forecast table
    for key in ["tfm", "moirai", "tirex", "tfmcov", "tfm3cov"]:
        model = re.match(r"zs_(.+?)(_multi)?(_ctx\d+d)?$", names[key]).group(1)
        flags = ["--tech", tech, "--days", "test"] + zs_flags(names[key])
        export = flags + ([] if model.endswith("-cov") else ["--no-covs"])
        env = ENVS / MODEL_ENV[model.removesuffix("-cov")] / "bin" / "python"
        # TimesFM 2.5's PyTorch code uses CUDA or the CPU, never the Apple GPU; Moirai runs on the CPU
        # (16_other_tsfms.py)
        device = CPU if model.removesuffix("-cov") in ("timesfm-2.5", "moirai-2.0") else GPU
        j[key] = (device, [("export windows", [PY, S / "16_other_tsfms.py", "export"] + export, True),
                        ("predict", [env, S / "16_other_tsfms.py", "predict", model] + flags, True),
                        ("collect", [PY, S / "16_other_tsfms.py", "collect", model] + flags, True)])
    # Fine-tuned Chronos-2 (15_finetune_chronos2.py): fine-tune (then it forecasts the validation half-year,
    # not counted), then forecast the test year with the saved model
    for key in ["c2ft", "c2ftElia"]:
        m = re.match(r"ft_chronos-2-(cov(?:-elia)?)(_multi)?_ctx(\d+)d_(lora|full)_lr([0-9.e-]+)_s(\d+)$", names[key])
        variant, multi, ctx, mode, lr, seed = m.groups()
        args = [PY, S / "15_finetune_chronos2.py", "--tech", tech, "--variant", variant, "--context-days", ctx,
                "--mode", mode, "--lr", lr, "--seed", seed, "--batch-size", "128" if ctx == "7" else "32"]
        args += ["--multi"] if multi else []
        j[key] = (GPU, [("fine-tune + validation forecast", args, False),
                        ("test forecast", args + ["--days", "test"], True)])
    return j


def run(cmd, env, cwd):
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
    return time.time() - t0


def inner_times(key, tech, scratch):
    """Times the scripts measure themselves: PG-Adapter training/inference, Chronos-2 fine-tuning."""
    rows = []
    for f in (scratch / "runs" / tech).glob("pga_*_s0.json"):
        info = json.loads(f.read_text())
        rows += [(f"{key}", "PG-Adapter training (own clock)", info["train_seconds"], False),
                 (f"{key}", "PG-Adapter test forecast (own clock)", info["infer_seconds"], False)]
    for f in (scratch / "models" / "chronos2_ft" / tech).glob("*/train_info.json"):
        rows.append((key, "fine-tuning (train_info.json)", 60 * json.loads(f.read_text())["minutes"], True))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", choices=["solar", "wind"], default="wind")
    ap.add_argument("--only", nargs="*", help="time these row keys only (default: all)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary results folder")
    ap.add_argument("--redo", action="store_true", help="time the --only models again, keeping the earlier "
                    "rows in timing_superseded.csv")
    a = ap.parse_args()
    if a.redo and not a.only:
        ap.error("--redo needs --only")
    names = table_names(a.tech)
    todo = jobs(a.tech, names)
    if a.only:
        todo = {k: v for k, v in todo.items() if k in a.only}
    out = RESULTS / "metrics" / "timing.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    machine = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True).stdout.strip(),
               "memory_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                               text=True).stdout) / 2 ** 30,
               "os": platform.platform(), "date": time.strftime("%Y-%m-%d")}
    (RESULTS / "metrics" / "timing_machine.json").write_text(json.dumps(machine, indent=1))
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["tech", "key"])
    if a.redo:
        old = (done["tech"] == a.tech) & done["key"].isin(todo)
        superseded = out.with_name("timing_superseded.csv")
        done[old].to_csv(superseded, mode="a", header=not superseded.exists(), index=False)
        done = done[~old]
        done.to_csv(out, index=False)
    rows = []
    for key, (device, steps) in todo.items():
        if ((done["tech"] == a.tech) & (done["key"] == key)).any():
            print(f"{key}: already timed, skipped", flush=True)
            continue
        # A fresh results folder per model, so that no model reuses another's cached windows or priors
        scratch = Path(tempfile.mkdtemp(prefix=f"pgtsfm_timing_{key}_"))
        env = dict(os.environ, PGTSFM_RESULTS=str(scratch), PYTHONWARNINGS="ignore")
        step_rows = []
        try:
            for step, cmd, counted in steps:
                secs = run(cmd, env, scratch)
                step_rows.append((key, step, secs, counted))
                print(f"{key}: {step} {secs:.0f} s", flush=True)
        except RuntimeError as e:  # report and go on with the next model; rerun later with --only
            print(f"{key}: FAILED\n{e}", flush=True)
            continue
        step_rows += inner_times(key, a.tech, scratch)
        rows += [dict(tech=a.tech, key=k, step=s, seconds=round(x, 1), counted=c, device=device,
                      file=names[key]) for k, s, x, c in step_rows]
        # Save after every model, so that an interrupted run keeps what it measured
        done = pd.concat([done, pd.DataFrame(rows[-len(step_rows):])], ignore_index=True)
        done.to_csv(out, index=False)
        if not a.keep:
            shutil.rmtree(scratch)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
