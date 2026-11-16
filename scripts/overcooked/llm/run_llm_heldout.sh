#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# GPT-6.1 (high-level actions, memory on) vs the NestRL held-out partners. 2026-11-04.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-pace}"
: "${LLM_BASE_URL:?set LLM_BASE_URL (OpenAI-compatible endpoint)}" "${LLM_API_KEY:?set LLM_API_KEY}"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
cd ${PACE_ROOT}
OUT=logs/Overcooked/llm_eval/gpt6_1_sol_high_heldout_20261104; mkdir -p $OUT
MAXJOBS=${MAXJOBS:-6}
run(){ local name=$1; shift
  [ -f $OUT/$name/summary.json ] && { echo "[skip] $name done"; return; }
  python eval_llm.py --model openai/gpt6_1_sol --actions high --memory full --test-pass 10 --test-episodes 5 --seed 1 \
    --out-dir $OUT/$name "$@" > $OUT/$name.log 2>&1
  echo "[done] $name rc=$? $(python -c "import json;d=json.load(open('$OUT/$name/summary.json'))['summary'];print('succ',d['success_rate'],'fail',d['llm']['failures'],'unpars',d['llm']['unparseable'])" 2>/dev/null)"; }
active=0
# FC: both-sides L1 pool seeds 6-10, LLM in seat 0 and seat 1
for s in 6 7 8 9 10; do for seat in 0 1; do
  run FC_L1_s${s}_seat${seat} --domain FC --env-config environment/overcooked/config/fc_small_test.yaml --player-id $seat \
      --partner-ckpt $PWD/logs/Overcooked/ppo_june_new_exp/L1_FC_bothsides_p025/seed${s}_seed${s}/ppo/30000000.pt &
  active=$((active+1)); [ $active -ge $MAXJOBS ] && { wait -n; active=$((active-1)); }; sleep 5
done; done
# FI: the 8 good held-out seeds of L1_FI_pace_p025_h80 (right seat), LLM in seat 0, H=80
for ts in R1R2:4 R1R2:7 R1R2:8 R3R4:5 R3R4:6 R3R4:9 R3R4:13 R5R6:6; do t=${ts%:*}; s=${ts#*:}
  run FI_L1_${t}_s${s} --domain FI --l0-pair-type $t --env-config environment/overcooked/config/fc_small_6recipe_h80.yaml --player-id 0 \
      --partner-ckpt $PWD/logs/Overcooked/june_new_exp/L1_FI_pace_p025_h80/right/$t/seed$s/ppo/30000000.pt &
  active=$((active+1)); [ $active -ge $MAXJOBS ] && { wait -n; active=$((active-1)); }; sleep 5
done
wait; echo "=== all done $(date '+%F %T')"
