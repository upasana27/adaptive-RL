#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Evaluate LEFT-side ego agents (baselines + FC L2) against HELD-OUT FC L1 partners.
#
# Egos (player-0):
#   - Baselines:  L2_left_agents/baselines/{PACE,LiAM,LiAMx,LiLi,Generalist}
#   - FC L2 pure: L2_left_agents/L2_left_big
#   - FC L2 Loft: june_new_exp/L2_Loft_FC/left/seed{1..4}
# Partners (player-1): held-out FC L1 right seeds (default 11,18,20 from L1_FC/right)
#                      — converged vs L0 and NOT used in FC L2 training.
#
# Protocol: 10 rounds of 5 episodes (--test-pass 10 --test-episodes 5), history
# clears every 5 (the training regime — NOT --pop-oldest-episode). Reuses
# online_test_.py + --partner-policy-paths (same mechanism as eval_LoftFC.sh).
#
# NOTE: LiAM / LiAMx were trained with --all-has-last-action (obs 79). As EGO they
# fail here (env gives 73) — the ego rollout doesn't append their last action.
# PACE/LiLi/Generalist and the FC L2 agents run fine. (The PARTNER side is already
# handled: PretrainedPolicy_test auto-appends last-action for such partners.)
#
# Output: logs/Overcooked/june_new_exp/eval_baselines_L2_vs_heldout_FC/{EGO}.log
# Summary: grep 'Test pass #9' <log> | grep success_rate_per_opponent
#
# Usage:
#   bash scripts/overcooked/june_new_exp/eval/left/eval_baselines_L2_vs_heldout_FC.sh
#   HELDOUT_SEEDS="11 18 20" GPUS="2 3 4 6" EGOS="PACE L2pureFC" bash .../eval_baselines_L2_vs_heldout_FC.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
cd ${PACE_ROOT}

HELDOUT_SEEDS=${HELDOUT_SEEDS:-"11 18 20"}
TEST_EPISODES=${TEST_EPISODES:-5}
TEST_PASS=${TEST_PASS:-10}
EGOS=${EGOS:-"PACE LiAM LiAMx LiLi Generalist L2pureFC L2loft1 L2loft2 L2loft3 L2loft4"}
GPUS=${GPUS:-"2 3 4 6"}
MAXJOBS=${MAXJOBS:-4}

OUT=logs/Overcooked/june_new_exp/eval_baselines_L2_vs_heldout_FC
mkdir -p "$OUT"

RB=logs/Overcooked/L2_left_agents/baselines
declare -A EGO
EGO[PACE]=$RB/PACE/ppo/30000000.pt
EGO[LiAM]=$RB/LiAM/ppo/30000000.pt
EGO[LiAMx]=$RB/LiAMx/ppo/30000000.pt
EGO[LiLi]=$RB/LiLi/ppo/30000000.pt
EGO[Generalist]=$RB/Generalist/ppo/30000000.pt
EGO[L2pureFC]=logs/Overcooked/L2_left_agents/L2_left_big/ppo/latest.pt
EGO[L2loft1]=logs/Overcooked/june_new_exp/L2_Loft_FC/left/seed1/ppo/latest.pt
EGO[L2loft2]=logs/Overcooked/june_new_exp/L2_Loft_FC/left/seed2/ppo/latest.pt
EGO[L2loft3]=logs/Overcooked/june_new_exp/L2_Loft_FC/left/seed3/ppo/latest.pt
EGO[L2loft4]=logs/Overcooked/june_new_exp/L2_Loft_FC/left/seed4/ppo/latest.pt

RBASE=logs/Overcooked/june_new_exp/L1_FC/right
PARTNERS=""
for s in $HELDOUT_SEEDS; do PARTNERS="${PARTNERS}${RBASE}/seed${s}/ppo/30000000.pt,"; done
PARTNERS="${PARTNERS%,}"
N_P=$(echo $HELDOUT_SEEDS | wc -w)

echo "=========================================================================="
echo "Baselines + FC L2  vs  held-out FC L1 [${HELDOUT_SEEDS}]"
echo "  Egos:     ${EGOS}"
echo "  Protocol: ${TEST_PASS} rounds x ${TEST_EPISODES} episodes (history clears every ${TEST_EPISODES})"
echo "  Out:      ${OUT}"
echo "=========================================================================="

run_one() {
  local name=$1 gpu=$2
  [ -f "${EGO[$name]}" ] || { echo "[skip] $name — missing ${EGO[$name]}"; return; }
  CUDA_VISIBLE_DEVICES=$gpu python online_test_.py --env-name Overcooked \
    --env_config './environment/overcooked/config/fc_small_test.yaml' \
    --algo ppo --exp-name "june_new_exp/eval_baselines_L2_vs_heldout_FC/${name}" --log-dir "$OUT/ld_${name}" \
    --player-id 0 --policy-dir "${EGO[$name]}" --seed 1 \
    --history-size 5 --include-current-episode --merge-encoder-computation \
    --train-pool-size ${N_P} --test-episodes ${TEST_EPISODES} --test-pass ${TEST_PASS} \
    --use-meta-episode --self-obs-mode \
    --partner-policy-paths "$PARTNERS" > "$OUT/${name}.log" 2>&1
  echo "[done] ${name} rc=$?"
}

read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
i=0; active=0
for name in $EGOS; do
  g=${GPU_ARR[$((i % NG))]}; i=$((i+1))
  run_one "$name" "$g" &
  active=$((active+1))
  if [ $active -ge $MAXJOBS ]; then wait -n; active=$((active-1)); fi
done
wait
echo "=========================================================================="
echo "Done. Summary:"
echo "  for f in ${OUT}/*.log; do echo \$f; grep 'Test pass #9' \$f | grep -oE \"success_rate_per_opponent.*\"; done"
echo "=========================================================================="
