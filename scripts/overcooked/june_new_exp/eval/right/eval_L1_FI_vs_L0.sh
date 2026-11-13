#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Evaluate FI L1 RIGHT agents against their own rule-based L0 pool.
#
# Uses online_test_.py (the established eval path) so success is computed the standard way
# (info['termination_info'].endswith('completed'), same as train_.py:613). Rule-based L0
# partners bypass the known-broken --partner-policy-paths route entirely.
#
# Protocol: 10 rounds x 5-episode meta-episodes (history clears every 5, matching training).
# ANCHOR (validated 2026-10-17): p=0.0 L1-left R1R2 seed1 @H=40 -> 0.717, matching its
# training-reported ~0.70. If a known pool stops reproducing its training score, the harness
# is wrong before the result is.
#
# Usage:
#   BASE=.../L1_FI_pace_p025_h80/right ENV_CFG=...fc_small_6recipe_h80.yaml P=0.25 \
#   TYPES="R1R2 R3R4 R5R6" SEEDS="1 2" GPUS="4 5 6" bash .../eval_L1_FI_vs_L0.sh
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
set -uo pipefail
REPO=${PACE_ROOT}; cd "$REPO"

BASE=${BASE:-$REPO/logs/Overcooked/june_new_exp/L1_FI_pace_p025_h80/right}
ENV_CFG=${ENV_CFG:-./environment/overcooked/config/fc_small_6recipe_h80.yaml}
P=${P:-0.25}
CKPT=${CKPT:-30000000.pt}
TYPES=${TYPES:-"R1R2 R3R4 R5R6"}
SEEDS=${SEEDS:-"1 2"}
PLAYER_ID=${PLAYER_ID:-1}
TEST_PASS=${TEST_PASS:-10}
TEST_EPISODES=${TEST_EPISODES:-5}
GPUS=${GPUS:-"4 5 6"}
OUT=${OUT:-logs/Overcooked/june_new_exp/eval_L1_FI_vs_L0}
mkdir -p "$OUT"
read -ra GPU_ARR <<< "$GPUS"; i=0

echo "=== FI L1 (player-id $PLAYER_ID) vs own L0 pool | p=$P | $TEST_PASS x $TEST_EPISODES | $(date '+%F %T') ==="
echo "    base=$BASE  cfg=$ENV_CFG  ckpt=$CKPT"
for t in $TYPES; do for s in $SEEDS; do
  ck="$BASE/$t/seed$s/ppo/$CKPT"
  if [ ! -f "$ck" ]; then echo "  [skip] $t/seed$s — no $CKPT"; continue; fi
  g=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}; i=$((i+1))
  lg="$OUT/${t}_seed${s}.log"
  CUDA_VISIBLE_DEVICES=$g python online_test_.py --env-name Overcooked \
    --env_config "$ENV_CFG" --algo ppo \
    --exp-name "june_new_exp/eval_L1_FI_vs_L0/${t}_seed${s}" --log-dir "$OUT/ld_${t}_s${s}" \
    --player-id $PLAYER_ID --policy-dir "$ck" --seed 1 \
    --latent-training --separate-history --self-obs-mode --use-meta-episode \
    --history-size 5 --include-current-episode --merge-encoder-computation \
    --test-episodes $TEST_EPISODES --test-pass $TEST_PASS \
    --l0-pair-type "$t" --p $P > "$lg" 2>&1
  sr=$(grep "Final result" "$lg" | grep -oE "'success_rate': \(0?\.?[0-9]+" | head -1 | grep -oE "[0-9.]+$")
  po=$(grep "Final result" "$lg" | grep -oE "success_rate_per_opponent': \(array\(\[[^]]*\]" | sed 's|.*array(\[||')
  rw=$(grep "Final result" "$lg" | grep -oE "'reward': \(-?[0-9.]+" | head -1 | grep -oE "\-?[0-9.]+$")
  printf "  %-14s success=%-8s reward=%-9s per-L0=[%s]\n" "$t/seed$s" "${sr:-ERR}" "${rw:-ERR}" "$po"
done; done
echo "=== done $(date '+%F %T') ==="
