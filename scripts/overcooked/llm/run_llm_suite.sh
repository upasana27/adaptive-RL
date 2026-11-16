#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Run the LLM-as-agent eval suite sequentially. One durable process, literal paths,
# unbuffered output (so a backgrounded run's log is readable while it runs).
#
# Usage:
#   MODEL=qwen36-35b-a3b PASSES=3 setsid nohup bash scripts/overcooked/llm/run_llm_suite.sh \
#     > logs/Overcooked/llm_eval/_suite.log 2>&1 < /dev/null &
#
# Cost: each episode = horizon calls. H=80 x 5 ep x PASSES passes = 400*PASSES calls
# per condition; at ~6 s/call that is ~40 min per pass per H=80 condition.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
: "${LLM_BASE_URL:?set LLM_BASE_URL (OpenAI-compatible endpoint)}" "${LLM_API_KEY:?set LLM_API_KEY}"
export PYTHONUNBUFFERED=1
set -uo pipefail
cd ${PACE_ROOT}

MODEL=${MODEL:-qwen36-35b-a3b}
PASSES=${PASSES:-3}
EPISODES=${EPISODES:-5}
SEAT=${SEAT:-0}
# plan-len >1 asks for a short action plan per call. Reasoning models are 5-7x slower
# on the real 1495-token state prompt, so plan-len is what makes them affordable
# (glm-5-3: 6 calls/episode instead of 40). Tradeoff: the plan executes WITHOUT
# re-observing, so the partner may move mid-plan - it trades coordination fidelity for cost.
PLAN_LEN=${PLAN_LEN:-1}
TAG=${TAG:-}
OUT=logs/Overcooked/llm_eval
mkdir -p "$OUT"
C6=environment/overcooked/config/fc_small_6recipe_h80.yaml
C3=environment/overcooked/config/fc_small_test.yaml
FIPOOL=${FIPOOL:-${PACE_ROOT}/logs/Overcooked/ppo_FI/right/L1/Ours}   # level-1 partner pool of the heterogeneous setting

run () {   # name, then extra args
  local name=$1; shift
  local dir="$OUT/${MODEL}${TAG}_${name}"
  if [ -f "$dir/summary.json" ]; then echo "[$(date '+%T')] SKIP $name (summary.json exists)"; return; fi
  echo "[$(date '+%T')] START $name"
  python eval_llm.py --model "$MODEL" --player-id "$SEAT" \
    --test-pass "$PASSES" --test-episodes "$EPISODES" --plan-len "$PLAN_LEN" \
    --out-dir "$dir" "$@" \
    > "$dir.log" 2>&1
  echo "[$(date '+%T')] DONE  $name rc=$? -> $(python3 -c "
import json,sys
try:
    s=json.load(open('$dir/summary.json'))['summary']
    print(f\"success={s['success_rate']:.3f} reward={s['mean_reward']:+.2f} ent={s['llm_action_entropy']} partner_ent={s['partner_action_entropy']} calls={s['llm']['calls']} unparse={s['llm']['unparseable']} trunc={s['llm'].get('truncated',0)}\")
except Exception as e: print('no summary:', e)" 2>/dev/null)"
}

echo "=== LLM suite: model=$MODEL${TAG} seat=player_$SEAT  $PASSES x $EPISODES  plan_len=$PLAN_LEN  $(date '+%F %T') ==="

# 1) vs rule-based L0 — FI domain (6-recipe disjoint), horizon 80, stochastic L0
run "FI_L0_R1R2_h80"  --domain FI --l0-pair-type R1R2 --env-config "$C6" --p 0.25
# 2) vs rule-based L0 — FC domain (3-recipe), horizon 40
run "FC_L0_R1_h40"    --domain FC --l0-pair-type R1   --env-config "$C3" --p 0.25
# 3) vs a trained L1 partner — FI, the new p=0.25/H=80 pool
run "FI_L1_R1R2s1_h80" --domain FI --env-config "$C6" \
    --partner-ckpt "$FIPOOL/R1R2/seed1/ppo/30000000.pt"
# 4) vs a trained L1 partner — a DIFFERENT type, to test partner identification
run "FI_L1_R5R6s1_h80" --domain FI --env-config "$C6" \
    --partner-ckpt "$FIPOOL/R5R6/seed1/ppo/30000000.pt"

echo "=== suite complete $(date '+%F %T') ==="
echo "--- summary table ---"
for d in "$OUT/${MODEL}${TAG}"_*/; do
  [ -f "$d/summary.json" ] || continue
  python3 -c "
import json
s=json.load(open('$d/summary.json'))['summary']
print(f\"  {'$d'.rstrip('/').split('/')[-1]:<44} success={s['success_rate']:.3f} reward={s['mean_reward']:+7.2f} llm_ent={s['llm_action_entropy']:<6} partner_ent={s['partner_action_entropy']}\")"
done
