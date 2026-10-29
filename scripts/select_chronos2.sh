#!/bin/zsh
# Chronos-2 zero-shot: settings chosen on the validation half-year, then the test year.
# 1. Validation runs: 7 or 28 days of power history, and ECMWF-only or ECMWF + ICON-EU + GFS weather
#    covariates, for each variant (power only, + weather, + weather and Elia's forecast).
# 2. Score them (results/metrics/diagnostics/chronos2_selection.csv).
# 3. Test year with the setting of lowest validation CRPS per technology and variant (choose_chronos2.py).
# Usage: zsh scripts/select_chronos2.sh > logs/select_chronos2.log 2>&1
cd "$(dirname "$0")/.."
ENVS=${PGTSFM_ENVS:-/opt/miniconda3/envs}  # folder of the conda environments (README); set PGTSFM_ENVS to change it
PY=$ENVS/pgtsfm/bin/python
run() { echo "[$(date +%H:%M)] $*"; $PY -W ignore "$@" 2>&1 | grep -v -E "Redirects|MallocStack|HF_TOKEN" | tail -1; }
have() { [[ -f results/forecasts/$1.parquet && -f results/forecasts/${1/solar/wind}.parquet ]]; }  # both techs

for ctx in 7 28; do
  sfx=$([[ $ctx == 7 ]] || echo "_ctx${ctx}d")
  have solar/selection/zs_chronos-2$sfx || run scripts/06_tsfm_priors.py chronos-2 --days val --context-days $ctx
  for bb in chronos-2-cov chronos-2-cov-elia; do
    have solar/selection/zs_$bb$sfx || run scripts/06_tsfm_priors.py $bb --days val --context-days $ctx
    have solar/selection/zs_${bb}_multi$sfx || run scripts/06_tsfm_priors.py $bb --days val --multi --context-days $ctx
  done
done
run scripts/13_check_collapse.py "zs_chronos-2*" --folder selection --metrics --out chronos2_selection

for line in "${(@f)$($PY scripts/choose_chronos2.py)}"; do
  set -- ${=line}  # tech backbone multi ctx
  name=zs_$2$([[ $3 == 1 ]] && echo _multi)$([[ $4 == 7 ]] || echo "_ctx$4d")
  [[ -f results/forecasts/$1/$name.parquet ]] ||
    run scripts/06_tsfm_priors.py $2 --days test --tech $1 $([[ $3 == 1 ]] && echo --multi) --context-days $4
done
echo "[$(date +%H:%M)] all done"
