#!/bin/zsh
# Experiment queue, run one job at a time (parallel jobs slow the laptop down badly).
# Usage: zsh scripts/run_queue.sh > logs/queue.log 2>&1
# Jobs whose forecast file already exists are skipped, so the queue can be stopped and restarted.
cd "$(dirname "$0")/.."
ENVS=${PGTSFM_ENVS:-/opt/miniconda3/envs}  # folder of the conda environments (README); set PGTSFM_ENVS to change it
PY=$ENVS/pgtsfm/bin/python
run() { echo "[$(date +%H:%M)] $*"; $PY -W ignore "$@" 2>&1 | grep -v -E "Redirects|MallocStack" | tail -1; }
have() { [[ -f results/forecasts/$1.parquet ]]; }  # e.g. have solar/pga_elia_365d_d2_s0

# 1. Main configurations with 3 seeds, then seed ensembles.
for tech in solar wind; do
  for bb in elia chronos-bolt-base none; do
    for multi in "" "--multi"; do
      for seed in 0 1 2; do
        have $tech/pga_${bb}${multi:+_multi}_365d_d2_s${seed} || run scripts/07_train_adapter.py --tech $tech --backbone $bb $multi --seed $seed
      done
      run scripts/10_seed_ensemble.py $tech "${bb}${multi:+_multi}_365d_d2"
    done
  done
done

# 2. Ablation: no physics layer (multi-model weather, Chronos prior).
for tech in solar wind; do
  for seed in 0 1 2; do
    have $tech/pga_chronos-bolt-base_nophys_multi_365d_d2_s${seed} || run scripts/07_train_adapter.py --tech $tech --backbone chronos-bolt-base --multi --no-physics --seed $seed
  done
  run scripts/10_seed_ensemble.py $tech "chronos-bolt-base_nophys_multi_365d_d2"
done

# 3. Limited data (E2): a new model at the start of every test month, trained on the 7-90 days before it.
for tech in solar wind; do
  for n in 7 14 30 90; do
    for bb in chronos-bolt-base none; do
      for seed in 0 1 2; do
        have $tech/pga_${bb}_multi_roll${n}d_d2_s${seed} || run scripts/07_train_adapter.py --tech $tech --backbone $bb --multi --train-days $n --rolling --seed $seed
      done
      run scripts/10_seed_ensemble.py $tech "${bb}_multi_roll${n}d_d2"
    done
    have $tech/lightgbm_multi_roll${n}d || run scripts/05_lightgbm.py d2 multi days=$n rolling tech=$tech
  done
done

# 4. Transfer to an unseen province (E3): leave one solar province out, 3 seeds each.
for prov in Antwerp Brussels East-Flanders Flemish-Brabant Hainaut Limburg Liège Luxembourg Namur Walloon-Brabant West-Flanders; do
  for seed in 0 1 2; do
    have solar/pga_chronos-bolt-base_multi_365d_d2_lo-${prov}_s${seed} || run scripts/07_train_adapter.py --tech solar --backbone chronos-bolt-base --multi --holdout $prov --seed $seed
  done
done
echo "[$(date +%H:%M)] transfer runs finished"

# 5. Trained deep-learning baselines (slowest). Settings were chosen on the validation half-year
#    with --select (results/forecasts/solar/selection/): global input scaling, 12 training windows
#    per day, and 1 day of history for N-HiTS. N-HiTS and TiDE run faster on the CPU.
for tech in solar wind; do
  have $tech/nf_nhits_multi_in1d || run scripts/09_neuralforecast.py $tech nhits --multi --input-days 1 --accelerator cpu
  have $tech/nf_tide_multi || run scripts/09_neuralforecast.py $tech tide --multi --accelerator cpu
  have $tech/nf_lstm_multi || run scripts/09_neuralforecast.py $tech lstm --multi
done
for tech in solar wind; do
  have $tech/nf_tft_multi || run scripts/09_neuralforecast.py $tech tft --multi
done

# 5b. Benchmark additions: LightGBM with Elia's forecast as input (Setting B), and Chronos-2 zero-shot
#     with and without covariates (settings chosen on the validation half-year inside the script).
for tech in solar wind; do
  have $tech/lightgbm_multi_elia || run scripts/05_lightgbm.py d2 multi elia tech=$tech
  have $tech/lightgbm_elia || run scripts/05_lightgbm.py d2 elia tech=$tech
done
zsh scripts/select_chronos2.sh

# 5c. TimesFM 2.5, Moirai 2.0 and TiRex zero-shot, and Chronos-2 fine-tuned (settings chosen on the
#     validation half-year by choose_settings.py). Needs the three model environments (README).
zsh scripts/run_more_tsfms.sh

# 5d. Extra row: TimesFM 3.0 with covariates, released during the test year (see run_timesfm3.sh).
zsh scripts/run_timesfm3.sh

# 6. Seed ensembles of every PG-Adapter configuration.
for f in results/forecasts/*/pga_*_s0.parquet; do
  tech=$(basename $(dirname $f)); tag=$(basename $f .parquet); tag=${tag#pga_}; tag=${tag%_s0}
  run scripts/10_seed_ensemble.py $tech $tag
done

# 6b. Compute cost (E7): every model in the main table timed on the wind series, one job at a time,
#     and its parameters counted.
#     Takes several hours; run it on an otherwise idle machine.
run scripts/17_timing.py
run scripts/18_parameters.py

# 7. Calibrated 80 % intervals (E5), imbalance cost (E6) and the comparison for the models in the paper's
#    tables, then the tables themselves, and a check that no PG-Adapter series-month collapsed to zero.
run scripts/14_make_tables.py --prepare
run scripts/13_check_collapse.py "pga_*"
echo "[$(date +%H:%M)] all done"
