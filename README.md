# Do time-series foundation models improve day-ahead solar and wind forecasts? A benchmark against Elia

Compares zero-shot foundation models (Chronos-Bolt, Chronos-2), a small trained network around a frozen
foundation model (PG-Adapter), gradient boosting and trained deep-learning models with the Belgian operator's
day-ahead forecast.

Research code for the paper described in [docs/research-design.md](docs/research-design.md).

## Setup (once)

```bash
conda create -y -n pgtsfm python=3.11
conda activate pgtsfm
pip install -r requirements.txt
conda install -y -c conda-forge llvm-openmp   # LightGBM needs OpenMP on macOS
```

`requirements.lock.txt` records the exact versions used for the reported results.

TimesFM, Moirai and TiRex need package versions that conflict with this environment (Moirai pins
torch 2.4 and numpy 1.26), so each has its own:

```bash
conda create -y -n tsfm-timesfm python=3.11 && conda run -n tsfm-timesfm pip install "timesfm[torch]==3.0.2" jax==0.10.2 scikit-learn
conda create -y -n tsfm-moirai python=3.11 && conda run -n tsfm-moirai pip install uni2ts==2.0.0
conda create -y -n tsfm-tirex python=3.11 && conda run -n tsfm-tirex pip install tirex-ts==1.4.2
```

(`timesfm[xreg]` would install the CUDA build of JAX; the CPU build is enough for its covariate regression.)
Exact versions: `envs/<env>.lock.txt`. The run scripts look for these environments in `/opt/miniconda3/envs`;
if yours are elsewhere, set `PGTSFM_ENVS` (for example `export PGTSFM_ENVS=~/miniconda3/envs`).

## Pipeline

Run the scripts in order from this folder. Each one saves its output, so later steps can be rerun on their own.

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_download_elia.py` | `data/raw/elia_{solar,wind,imbalance}.parquet` (Elia Open Data) |
| 2 | `scripts/02_download_nwp.py` | `data/raw/nwp_<lat>_<lon>.parquet` (archived ECMWF forecasts via Open-Meteo) |
| 3 | `scripts/03_build_dataset.py` | `data/processed/{solar,wind}.parquet` (15-min tables) |
| 4 | `scripts/04_simple_baselines.py` | persistence, 7-day climatology, Elia day-ahead forecast |
| 5 | `scripts/05_lightgbm.py` | LightGBM quantile baseline (`elia` adds Elia's forecast as input; `days=N rolling` for limited data) |
| 6 | `scripts/06_tsfm_priors.py` | zero-shot foundation-model forecasts and PG-Adapter priors (`chronos-2-cov[-elia]` for covariates) |
| 7 | `scripts/07_train_adapter.py` | PG-Adapter (`--train-days N --rolling` for limited data, `--holdout <province>` for transfer) |
| 8 | `scripts/08_compare.py` | test-year metrics and Diebold–Mariano tests (`results/metrics/<tech>_compare.csv`) |
| 9 | `scripts/09_neuralforecast.py` | N-HiTS, TiDE, TFT, LSTM baselines (`--select` chooses settings on the validation half-year) |
| 10 | `scripts/10_seed_ensemble.py` | average of the 3 PG-Adapter seeds (`*_ens`) |
| 11 | `scripts/11_value.py` | imbalance cost with real Belgian prices |
| 12 | `scripts/12_conformal.py` | adaptive conformal calibration of the 80 % interval (`*_aci`) |
| 13 | `scripts/13_check_collapse.py` | counts series-months forecast as all zero (`results/metrics/diagnostics/`) |
| 14 | `scripts/14_make_tables.py` | the paper's tables, figure and number macros (`paper/tables/`, `paper/figures/`, `paper/results_macros.tex`; `--prepare` first redoes steps 11, 12 and 8 for the models in the tables) |
| 15 | `scripts/15_finetune_chronos2.py` | Chronos-2 fine-tuned (LoRA) on the training year (`results/models/chronos2_ft/`) |
| 16 | `scripts/16_other_tsfms.py` | TimesFM 2.5, TimesFM 3.0, Moirai 2.0 and TiRex zero-shot (`export` and `collect` in this environment, `predict` in the model's own, see below) |
| 17 | `scripts/17_timing.py` | compute cost: reruns every model in the main table once on the wind series into a temporary folder and times it (`results/metrics/timing.csv`) |
| 18 | `scripts/18_parameters.py` | parameter counts of the same models, total and trained (`results/metrics/parameters.csv`) |

`scripts/run_queue.sh` runs steps 5–14 for every configuration in the paper, one job at a time, and skips
jobs whose forecast file already exists. `scripts/select_unclipped.sh` is the check, on the months before the
test year, that chose PG-Adapter's training loss; `scripts/select_chronos2.sh` chooses Chronos-2's settings
on the validation half-year. `scripts/run_more_tsfms.sh` adds TimesFM 2.5, Moirai 2.0, TiRex and fine-tuned
Chronos-2, with settings chosen on the validation half-year by `scripts/choose_settings.py`.
`scripts/run_timesfm3.sh` adds TimesFM 3.0 with covariates, an extra row because it was released during the
test year. Its weights (`google/timesfm-3.0-pytorch`, 1.3 GB) are under the TimesFM Non-Commercial License
v1.0, which you accept by downloading them.

Step 14 writes the paper's tables, figure and number macros to `paper/`. The manuscript itself will be added to
this repository once it is accepted; until then, every number in it can be checked against the metric tables in
`results/metrics/`.

Forecasts are saved in `results/forecasts/<tech>/<model>.parquet`; metric tables in `results/metrics/`.
Subfolders hold runs that are not part of the comparison: `selection/` (validation-only runs used to choose
deep-learning settings), `diagnostics/` and `superseded/` (kept for the record, see
`results/forecasts/README_superseded.md`).

## Forecast setting

- Issued at 09:00 UTC on day D-1 for day D (96 quarter-hours, 00:00–23:45 UTC).
- Inputs: the last 7 days of observed power, plus weather forecasts for day D that were already available at issue time.
- Errors are in % of monitored capacity.

## Code layout

- `src/pgtsfm/config.py`: study period, splits, regions and weather points
- `src/pgtsfm/windows.py`: turns the tables into day-ahead samples
- `src/pgtsfm/features.py`: weather, physics, calendar and recent-power features
- `src/pgtsfm/metrics.py`: nMAE, nRMSE, CRPS, coverage, Diebold–Mariano test

## Licence and data

The code is released under the MIT licence (`LICENSE`). The data are not redistributed here: the scripts download
them from Elia Open Data, Open-Meteo and Energy-Charts, whose own terms apply. Model weights are downloaded from
their publishers; TimesFM 3.0's are under the TimesFM Non-Commercial License. To cite this code, see
`CITATION.cff`.
