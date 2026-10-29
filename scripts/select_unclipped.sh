#!/bin/zsh
# Choosing the training loss without the test year: rolling monthly retraining on the 12 months
# before it (--dev), with the loss on clipped quantiles (first version, *_clipped_*) and on unclipped
# ones (now the default). Run on 2026-10-26; the clipped loss collapsed 421 of 1,728 series-months
# to an all-zero forecast, the unclipped loss none. (These runs were made when the unclipped loss
# was an option, so their JSON files record "unclipped_loss" instead of "clipped_loss".)
# Usage: zsh scripts/select_unclipped.sh > logs/select_unclipped.log 2>&1
# Then:  python scripts/13_check_collapse.py "pga_*roll*" --folder selection --metrics
cd "$(dirname "$0")/.."
ENVS=${PGTSFM_ENVS:-/opt/miniconda3/envs}  # folder of the conda environments (README); set PGTSFM_ENVS to change it
PY=$ENVS/pgtsfm/bin/python
run() { echo "[$(date +%H:%M)] $*"; $PY -W ignore "$@" 2>&1 | grep -v -E "Redirects|MallocStack" | tail -1; }
have() { [[ -f results/forecasts/$1.parquet ]]; }

for tech in wind solar; do
  for n in 7 14; do
    for bb in none chronos-bolt-base; do
      [[ $bb == chronos-bolt-base && $n == 14 ]] && continue  # the prior does not change the output head
      for seed in 0 1 2; do
        have $tech/selection/pga_${bb}_multi_roll${n}d_d2_clipped_s${seed} ||
          run scripts/07_train_adapter.py --tech $tech --backbone $bb --multi --train-days $n --rolling --dev --clipped-loss --seed $seed
        have $tech/selection/pga_${bb}_multi_roll${n}d_d2_s${seed} ||
          run scripts/07_train_adapter.py --tech $tech --backbone $bb --multi --train-days $n --rolling --dev --seed $seed
      done
    done
  done
done
echo "[$(date +%H:%M)] all done"
