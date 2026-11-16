#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# LLM ego (high-level skills, round memory) against the held-out level-1 partners, 10 rounds x 5 episodes.
# Variables: MODEL (openai/gpt6_1_sol), FC_POOL, FI_POOL (level-1 pools), FC_SEEDS, FI_PARTNERS ("TYPE:seed ..."), MAXJOBS.
# Needs LLM_BASE_URL and LLM_API_KEY for an OpenAI-compatible endpoint.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-pace}"
: "${LLM_BASE_URL:?set LLM_BASE_URL (OpenAI-compatible endpoint)}" "${LLM_API_KEY:?set LLM_API_KEY}"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
cd ${PACE_ROOT}
MODEL=${MODEL:-openai/gpt6_1_sol}
FC_POOL=${FC_POOL:-$PWD/logs/Overcooked/ppo_FC/bothsides/L1/Ours}; FI_POOL=${FI_POOL:-$PWD/logs/Overcooked/ppo_FI/right/L1/Ours}
FC_SEEDS=${FC_SEEDS:-"16 17 18 19 20 21 22 23 24 25"}
FI_PARTNERS=${FI_PARTNERS:-"$(for t in R1R2 R3R4 R5R6; do for s in 16 17 18 19 20 21 22 23 24 25; do printf "%s:%s " $t $s; done; done)"}
ckpt() { for d in "$1/seed$2" "$1/seed$2_seed$2"; do [ -f "$d/ppo/30000000.pt" ] && { echo "$d/ppo/30000000.pt"; return; }; done; echo "$1/seed$2/ppo/30000000.pt"; }
OUT=${OUT:-logs/Overcooked/llm_eval/$(basename "$MODEL")_heldout}; mkdir -p $OUT
MAXJOBS=${MAXJOBS:-6}
run(){ local name=$1; shift
  [ -f $OUT/$name/summary.json ] && { echo "[skip] $name done"; return; }
  python eval_llm.py --model "$MODEL" --actions high --memory full --test-pass 10 --test-episodes 5 --seed 1 \
    --out-dir $OUT/$name "$@" > $OUT/$name.log 2>&1
  echo "[done] $name rc=$? $(python -c "import json;d=json.load(open('$OUT/$name/summary.json'))['summary'];print('succ',d['success_rate'],'fail',d['llm']['failures'],'unpars',d['llm']['unparseable'])" 2>/dev/null)"; }
active=0
# FC: held-out partners of the both-sides level-1 pool, LLM in seat 0 and seat 1
for s in $FC_SEEDS; do for seat in 0 1; do
  run FC_L1_s${s}_seat${seat} --domain FC --env-config environment/overcooked/config/fc_small_test.yaml --player-id $seat \
      --partner-ckpt "$(ckpt $FC_POOL $s)" &
  active=$((active+1)); [ $active -ge $MAXJOBS ] && { wait -n; active=$((active-1)); }; sleep 5
done; done
# FI: held-out partners of the right-seat level-1 pool, LLM in seat 0, H=80
for ts in $FI_PARTNERS; do t=${ts%:*}; s=${ts#*:}
  run FI_L1_${t}_s${s} --domain FI --l0-pair-type $t --env-config environment/overcooked/config/fc_small_6recipe_h80.yaml --player-id 0 \
      --partner-ckpt "$(ckpt $FI_POOL/$t $s)" &
  active=$((active+1)); [ $active -ge $MAXJOBS ] && { wait -n; active=$((active-1)); }; sleep 5
done
wait; echo "=== all done $(date '+%F %T')"
