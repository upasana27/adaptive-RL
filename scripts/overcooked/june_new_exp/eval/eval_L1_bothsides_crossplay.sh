#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Cross-play matrix for a BOTH-SIDES L1 pool: ego seed i (LEFT, player-id 0) x partner seed j
# (RIGHT), all i,j incl. the diagonal (self-play). 10x5 meta-episode protocol, partner history fed.
# Purpose: diversity check -- if every pair plays well together the seeds are near-identical
# conventions; low off-diagonal success = behaviourally diverse partners.
# Invocation mirrors eval_baselines_L2_vs_heldout_FC.sh (validated: FC control reproduces 0.8333).
#
# Usage: POOL=L1_FC_bothsides_p025 SEEDS="1 2 3 4 5" GPUS="4 5 6 7" bash .../eval_L1_bothsides_crossplay.sh
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REPO=${PACE_ROOT}; cd "$REPO"
POOL=${POOL:-L1_FC_bothsides_p025}
SEEDS=${SEEDS:-"1 2 3 4 5"}
ENV_CFG=${ENV_CFG:-./environment/overcooked/config/fc_small_test.yaml}
CKPT=${CKPT:-30000000.pt}
TEST_PASS=${TEST_PASS:-10}; TEST_EPISODES=${TEST_EPISODES:-5}
GPUS=${GPUS:-"4 5 6 7"}; MAXJOBS=${MAXJOBS:-4}; NICE=${NICE:-10}
OUT=${OUT:-logs/Overcooked/june_new_exp/eval_${POOL}_crossplay}
mkdir -p "$OUT"
ck() { local s=$1; for d in logs/Overcooked/june_new_exp/$POOL/seed$s/ppo logs/Overcooked/ppo_june_new_exp/$POOL/seed${s}_seed${s}/ppo; do [ -f $d/$CKPT ] && { echo $d/$CKPT; return; }; done; }
read -ra G <<< "$GPUS"; i=0
echo "=== cross-play $POOL  ego=LEFT x partner=RIGHT  ${TEST_PASS}x${TEST_EPISODES}  $(date '+%F %T') ==="
for e in $SEEDS; do for p in $SEEDS; do
  E=$(ck $e); P=$(ck $p); [ -z "$E" ] || [ -z "$P" ] && { echo "skip $e x $p (missing ckpt)"; continue; }
  g=${G[$((i % ${#G[@]}))]}; i=$((i+1)); cell="ego${e}_par${p}"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  (
  CUDA_VISIBLE_DEVICES=$g nice -n $NICE python online_test_.py --env-name Overcooked --env_config "$ENV_CFG" \
    --algo ppo --exp-name "june_new_exp/eval_${POOL}_crossplay/$cell" --log-dir "$OUT/ld_$cell" \
    --player-id 0 --policy-dir "$E" --seed 1 \
    --history-size 5 --include-current-episode --merge-encoder-computation \
    --train-pool-size 1 --test-episodes $TEST_EPISODES --test-pass $TEST_PASS \
    --use-meta-episode --self-obs-mode --feed-partner-history \
    --partner-policy-paths "$P" > "$OUT/$cell.log" 2>&1
  sr=$(grep "Final result" "$OUT/$cell.log" | grep -oE "'success_rate': \([0-9.]+" | head -1 | grep -oE "[0-9.]+$")
  echo "  $cell success=${sr:-ERR}"
  ) &
done; done
wait
echo "=== done $(date '+%F %T') ==="
