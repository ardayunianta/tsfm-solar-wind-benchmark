"""Parameter counts of every model in the main table (experiment E7, next to the times from 17_timing.py).

Total: the parameters used to make a forecast, sum(p.numel() for p in model.parameters()).
Trained: the parameters fitted on our data. Zero-shot models train none (TimesFM 2.5 with weather fits a linear
regression on the covariates of each forecast window, which is not counted). Fine-tuned Chronos-2 trains its LoRA
adapter, counted from the saved checkpoint. PG-Adapter's count was saved by 07_train_adapter.py; with the
Chronos-Bolt prior, the frozen backbone adds to its total. The neural baselines are built with the settings of
their reported forecasts (09_neuralforecast.make_model) and counted untrained. LightGBM is a tree ensemble and has
no comparable count.

The foundation models are loaded as in 06_tsfm_priors.py and 16_other_tsfms.py. TimesFM, Moirai and TiRex live in
their own environments (README), so the script calls itself there with --count <model>.

Usage: python scripts/18_parameters.py [--tech wind]
Output: results/metrics/parameters.csv
"""
import argparse
import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
# Rows of the main table -> the foundation model they run (fine-tuned Chronos-2 keeps the size of Chronos-2)
TSFM = {"bolt": "chronos-bolt-base", "c2": "chronos-2", "c2cov": "chronos-2", "c2covElia": "chronos-2",
        "tfm": "timesfm-2.5", "tfmcov": "timesfm-2.5", "moirai": "moirai-2.0", "tirex": "tirex",
        "tfm3cov": "timesfm-3.0", "c2ft": "chronos-2", "c2ftElia": "chronos-2"}
CHRONOS = {"chronos-bolt-base": "amazon/chronos-bolt-base", "chronos-2": "amazon/chronos-2"}


def script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), S / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def n_params(obj, trainable_only=False):
    """Parameters of the torch modules in obj, which is either a model or one of the wrapper objects the
    packages return (searched a few attributes deep). Each parameter is counted once."""
    import torch
    seen, found, todo = {}, set(), [obj]
    for _ in range(3):
        deeper = []
        for o in todo:
            if isinstance(o, torch.nn.Module):
                found.add(type(o).__name__)
                seen.update((id(p), p.numel()) for p in o.parameters() if p.requires_grad or not trainable_only)
            elif hasattr(o, "__dict__"):
                deeper += list(vars(o).values())
        todo = deeper
    return sum(seen.values()), sorted(found)


def load_other(model):
    """A foundation model from its own environment, loaded as in 16_other_tsfms.py (on the CPU)."""
    if model == "tirex":
        from tirex import load_model
        return load_model("NX-AI/TiRex", backend="torch", device="cpu")
    if model == "timesfm-2.5":
        import timesfm
        return timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    if model == "timesfm-3.0":
        from timesfm3 import TimesFM3Evaluator
        return TimesFM3Evaluator.from_pretrained("google/timesfm-3.0-pytorch", device="cpu", per_core_batch_size=16,
                                                 revision="43046b85ec22d584a13f8098c2ed39c889e129c2")
    if model == "moirai-2.0":
        from uni2ts.model.moirai2 import Moirai2Module
        return Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    raise ValueError(model)


def safetensors_size(path):
    """Number of values in a .safetensors file, read from its header (no weights are loaded)."""
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    return sum(math.prod(t["shape"]) for t in header.values())


def tsfm_size(model, timing):
    if model in CHRONOS:
        import torch
        from chronos import BaseChronosPipeline
        n, found = n_params(BaseChronosPipeline.from_pretrained(CHRONOS[model], device_map="cpu",
                                                                torch_dtype=torch.float32))
    else:
        env = timing.ENVS / timing.MODEL_ENV[model] / "bin" / "python"
        out = subprocess.run([env, "-W", "ignore", __file__, "--count", model], capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(out.stderr[-2000:])
        n, found = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"{model}: {n:,} parameters in {', '.join(found)}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", choices=["solar", "wind"], default="wind")
    ap.add_argument("--count", help="(internal) count one foundation model in its own environment")
    a = ap.parse_args()
    if a.count:
        print(json.dumps(n_params(load_other(a.count))))
        return

    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    from pgtsfm.config import RESULTS
    from pgtsfm.features import extra_cols
    from pgtsfm.windows import CONTEXT_STEPS
    timing, nf = script("17_timing.py"), script("09_neuralforecast.py")
    names = timing.table_names(a.tech)
    sizes = {m: tsfm_size(m, timing) for m in dict.fromkeys(TSFM.values())}
    lora = {k: safetensors_size(RESULTS / "models" / "chronos2_ft" / a.tech / names[k] / "finetuned-ckpt"
                                / "adapter_model.safetensors") for k in ["c2ft", "c2ftElia"]}
    rows = [dict(key=k, total=sizes[m], trained=lora.get(k, 0)) for k, m in TSFM.items()]
    for k in ["nhits", "tide", "tft", "lstm"]:
        name = names[k]
        days = re.search(r"_in(\d+)d", name)
        exog = nf.BASE_EXOG + (extra_cols("d2") if "_multi" in name else [])
        model = nf.make_model(k, exog, 1500, 0, "cpu", 8, "global", int(days.group(1)) if days else CONTEXT_STEPS // 96)
        n, _ = n_params(model, trainable_only=True)  # the loss keeps its quantile levels as frozen parameters
        rows.append(dict(key=k, total=n, trained=n))
    for k in ["pgaNone", "pgaBolt", "pgaElia"]:
        info = json.loads((RESULTS / "runs" / a.tech / f"{names[k].removesuffix('_ens')}_s0.json").read_text())
        rows.append(dict(key=k, total=info["n_params"] + (sizes["chronos-bolt-base"] if k == "pgaBolt" else 0),
                         trained=info["n_params"]))
    rows += [dict(key=k, total=float("nan"), trained=float("nan")) for k in ["lgb", "lgbElia"]]
    out = RESULTS / "metrics" / "parameters.csv"
    pd.DataFrame(rows).assign(tech=a.tech)[["tech", "key", "total", "trained"]].to_csv(out, index=False)
    print(pd.DataFrame(rows).set_index("key").to_string())
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
