#!/bin/zsh
# Extra row: TimesFM 3.0 zero-shot with weather covariates (16_other_tsfms.py, model timesfm-3.0-cov).
# It was released in August 2026, during the test year, and added after the other test results were known
# (docs/research-design.md). Same grid and selection rule as the other zero-shot models: 7 or 28 days of
# history, ECMWF only or all three weather models, lowest CRPS on the validation half-year (choose_settings.py).
# Uses the Apple GPU: run it after run_more_tsfms.sh, not alongside the fine-tuning.
# Finished runs are skipped, so the script can be stopped and restarted.
# Usage: zsh scripts/run_timesfm3.sh > logs/timesfm3.log 2>&1
cd "$(dirname "$0")/.."
ENVS=${PGTSFM_ENVS:-/opt/miniconda3/envs}  # folder of the conda environments (README); set PGTSFM_ENVS to change it
PY=$ENVS/pgtsfm/bin/python
TFM=$ENVS/tsfm-timesfm/bin/python
run() { echo "[$(date +%H:%M)] ${@:2}"; "$@" 2>&1 | grep -v -E "Redirects|MallocStack|HF_TOKEN|[Ww]arning" | tail -1; }

# One run: zs <tech> <val|test> <multi 0|1> <context days>
zs() {
  local tech=$1 split=$2 multi=$3 ctx=$4 cov=ecmwf flags=(--tech $1 --days $2 --context-days $4)
  [[ $multi == 1 ]] && { cov=multi; flags+=(--multi) }
  local n=zs_timesfm-3.0-cov$([[ $multi == 1 ]] && echo _multi)$([[ $ctx == 7 ]] || echo "_ctx${ctx}d")
  [[ -f results/forecasts/$tech$([[ $split == val ]] && echo /selection)/$n.parquet ]] && return
  [[ -f results/cache/windows/$tech/${split}_ctx${ctx}d_$cov.npz ]] ||
    run $PY -W ignore scripts/16_other_tsfms.py export $flags
  run $TFM -W ignore scripts/16_other_tsfms.py predict timesfm-3.0-cov $flags
  run $PY -W ignore scripts/16_other_tsfms.py collect timesfm-3.0-cov $flags
}

# 1. Validation half-year, then the scores of all zero-shot runs (the other models' choices do not change).
for tech in wind solar; do
  for ctx in 7 28; do
    for multi in 0 1; do zs $tech val $multi $ctx; done
  done
done
run $PY -W ignore scripts/13_check_collapse.py "zs_[tm]*" --folder selection --metrics --out other_tsfm_selection

# 2. Test year with the chosen settings.
for line in "${(@f)$($PY scripts/choose_settings.py zs)}"; do
  set -- ${=line}  # zs tech model multi ctx
  [[ $3 == timesfm-3.0-cov ]] && zs $2 test $4 $5
done
echo "[$(date +%H:%M)] timesfm-3.0 done"
