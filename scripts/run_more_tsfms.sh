#!/bin/zsh
# Benchmark additions: TimesFM 2.5, Moirai 2.0 and TiRex zero-shot (16_other_tsfms.py), and Chronos-2
# fine-tuned on the training year (15_finetune_chronos2.py). One job at a time, as in run_queue.sh.
# Settings are chosen on the validation half-year (choose_settings.py), then run on the test year.
# Finished runs are skipped, so the queue can be stopped and restarted.
# Usage: zsh scripts/run_more_tsfms.sh > logs/more_tsfms.log 2>&1
cd "$(dirname "$0")/.."
ENVS=${PGTSFM_ENVS:-/opt/miniconda3/envs}  # folder of the conda environments (README); set PGTSFM_ENVS to change it
PY=$ENVS/pgtsfm/bin/python
run() { echo "[$(date +%H:%M)] ${@:2}"; "$@" 2>&1 | grep -v -E "Redirects|MallocStack|HF_TOKEN|[Ww]arning" | tail -1; }
envpy() { case $1 in tirex) echo $ENVS/tsfm-tirex/bin/python;; timesfm*) echo $ENVS/tsfm-timesfm/bin/python;;
                     moirai*) echo $ENVS/tsfm-moirai/bin/python;; esac; }

# One zero-shot run: zs <model> <tech> <val|test> <multi 0|1> <context days>
zs() {
  local m=$1 tech=$2 split=$3 multi=$4 ctx=$5
  local n=zs_$m$([[ $multi == 1 ]] && echo _multi)$([[ $ctx == 7 ]] || echo "_ctx${ctx}d")
  [[ -f results/forecasts/$tech$([[ $split == val ]] && echo /selection)/$n.parquet ]] && return
  local cov=none flags=(--tech $tech --days $split --context-days $ctx)
  if [[ $m == *-cov ]]; then
    cov=$([[ $multi == 1 ]] && echo multi || echo ecmwf)
    [[ $multi == 1 ]] && flags+=(--multi)
  fi
  [[ -f results/cache/windows/$tech/${split}_ctx${ctx}d_$cov.npz ]] ||
    run $PY -W ignore scripts/16_other_tsfms.py export $flags $([[ $cov == none ]] && echo --no-covs)
  run $(envpy $m) -W ignore scripts/16_other_tsfms.py predict $m $flags
  run $PY -W ignore scripts/16_other_tsfms.py collect $m $flags
}

# 1. Zero-shot on the validation half-year: 7 or 28 days of history; with covariates, ECMWF only or
#    all three weather models. Moirai 2.0 gets no covariates (see 16_other_tsfms.py).
for tech in wind solar; do
  for ctx in 7 28; do
    for m in tirex timesfm-2.5 moirai-2.0; do zs $m $tech val 0 $ctx; done
    for multi in 0 1; do zs timesfm-2.5-cov $tech val $multi $ctx; done
  done
done
run $PY -W ignore scripts/13_check_collapse.py "zs_[tm]*" --folder selection --metrics --out other_tsfm_selection

# 2. Zero-shot on the test year with the chosen settings.
for line in "${(@f)$($PY scripts/choose_settings.py zs)}"; do
  set -- ${=line}  # zs tech model multi ctx
  zs $3 $2 test $4 $5
done
echo "[$(date +%H:%M)] zero-shot done"

# 3. Chronos-2 fine-tuned with LoRA on the training year, for weather covariates (Setting A) and weather
#    plus Elia's forecast (Setting B). Grid: learning rate 1e-5, 1e-4 or 3e-4; 7 days of history with all
#    three weather models (128 rows per batch), or 28 days with ECMWF only (32 rows per batch). Longer history
#    with all three models, larger batches and full fine-tuning do not fit in this laptop's 16 GB.
#    3e-4 was added on 2026-10-28, after 1e-4 won all eight validation comparisons and before any fine-tuned
#    test forecast was scored (docs/research-design.md). Its log is logs/more_tsfms_lr3e-4.log.
for variant in cov cov-elia; do
  for tech in wind solar; do
    for lr in 1e-5 1e-4 3e-4; do
      run $PY -W ignore scripts/15_finetune_chronos2.py --tech $tech --variant $variant --multi --context-days 7 \
          --mode lora --lr $lr --batch-size 128
      run $PY -W ignore scripts/15_finetune_chronos2.py --tech $tech --variant $variant --context-days 28 \
          --mode lora --lr $lr --batch-size 32
    done
  done
done
run $PY -W ignore scripts/13_check_collapse.py "ft_*" --folder selection --metrics --out finetune_selection

# 4. Fine-tuned models on the test year with the chosen settings.
for line in "${(@f)$($PY scripts/choose_settings.py ft)}"; do
  set -- ${=line}  # ft tech variant multi ctx mode lr seed
  run $PY -W ignore scripts/15_finetune_chronos2.py --tech $2 --variant $3 $([[ $4 == 1 ]] && echo --multi) \
      --context-days $5 --mode $6 --lr $7 --seed $8 --batch-size $([[ $5 == 7 ]] && echo 128 || echo 32) --days test
done
echo "[$(date +%H:%M)] all done"
