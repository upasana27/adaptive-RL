#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Generic L1 (FC, multi-recipe) vs L0 eval, with trajectory saving per seed.
#
# Reusable for any L1_FC checkpoint pool — defaults to L1_right_p025 (the
# Loft_FC training/eval pool) but you can override L1_BASE_DIR.
#
# Output layout:
#   logs/Overcooked/june_new_exp/eval_L1_FC_vs_L0/
#     ├── seed{N}/
#     │   ├── eval.log                       (online_test_ stdout)
#     │   └── trajectories/final/
#     │       └── opponent_{0,1,2}_trajectories.pkl   (one per L0 partner)
#
# L0 pool order (verified in env): partner 0 = L0(LO), partner 1 = L0(TC),
# partner 2 = L0(PB).  Wait — confirm via the recipe distribution
# in trajectories; the order can shift if --pool-seed changes.
#
# Usage:
#   bash scripts/overcooked/june_new_exp/eval_L1_FC_vs_L0.sh
#   SEEDS="11 18 20"               bash ...                  # default
#   SEEDS="11 18 19 20" GPU=0      bash ...
#   TEST_EPISODES=100              bash ...
#   L1_BASE_DIR=...                bash ...                  # custom pool

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

# Defaults: the seeds that matter for the Loft_FC analysis.
#   11, 18, 20 are the truly-held-out L1s used as Loft_FC final test partners
#   19, 7 are top performers from the Loft_FC training pool (for comparison)
SEEDS=${SEEDS:-"11 18 19 20"}
GPU=${GPU:-0}
TEST_EPISODES=${TEST_EPISODES:-50}
# SIDE = "right" (default) or "left" — selects the L1 pool side AND player-id.
SIDE=${SIDE:-right}
if [[ "${SIDE}" == "left" ]]; then
    PLAYER_ID=0
    L1_BASE_DIR=${L1_BASE_DIR:-${PACE_ROOT}/logs/Overcooked/L1_agents/L1_left_p025}
    OUT_ROOT=logs/Overcooked/june_new_exp/eval_L1_FC_LEFT_vs_L0
else
    PLAYER_ID=1
    L1_BASE_DIR=${L1_BASE_DIR:-${PACE_ROOT}/logs/Overcooked/L1_agents/L1_right_p025}
    OUT_ROOT=logs/Overcooked/june_new_exp/eval_L1_FC_vs_L0
fi
mkdir -p "${OUT_ROOT}"

echo "=========================================================================="
echo "L1_FC vs L0 evaluation"
echo "  L1 seeds:   ${SEEDS}"
echo "  L1 base:    ${L1_BASE_DIR}"
echo "  Episodes:   ${TEST_EPISODES} per L0 partner per L1 seed"
echo "  GPU:        ${GPU}"
echo "  Out root:   ${OUT_ROOT}"
echo "=========================================================================="

for SEED in ${SEEDS}; do
    L1_CKPT="${L1_BASE_DIR}/seed${SEED}/ppo/30000000.pt"
    if [ ! -f "${L1_CKPT}" ]; then
        echo "SKIP seed ${SEED}: L1 checkpoint not found at ${L1_CKPT}"
        continue
    fi

    OUT_DIR="${OUT_ROOT}/seed${SEED}"
    LOG_DIR="${OUT_DIR}/log_internal"
    TRAJ_DIR="${OUT_DIR}/trajectories"
    LOG_FILE="${OUT_DIR}/eval.log"
    mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TRAJ_DIR}"

    echo ""
    echo "=========================================================================="
    echo "L1 seed ${SEED}  vs  3 L0 rule-based partners"
    echo "  Checkpoint:   ${L1_CKPT}"
    echo "  Trajectories: ${TRAJ_DIR}/final/"
    echo "  Stdout log:   ${LOG_FILE}"
    echo "=========================================================================="

    CUDA_VISIBLE_DEVICES=${GPU} python online_test_.py --env-name Overcooked \
      --env_config './environment/overcooked/config/fc_small_test.yaml' \
      --recipe-type cross \
      --algo ppo \
      --exp-name "june_new_exp/eval_L1_FC_vs_L0/seed${SEED}" \
      --log-dir "${LOG_DIR}" \
      --player-id ${PLAYER_ID} \
      --policy-dir "${L1_CKPT}" \
      --seed ${SEED} \
      --history-size 5 \
      --include-current-episode \
      --merge-encoder-computation \
      --train-pool-size 3 \
      --eval-pool-size 0 \
      --rule-based-opponents 3 \
      --test-episodes ${TEST_EPISODES} \
      --pop-oldest-episode \
      --use-meta-episode \
      --self-obs-mode \
      --separate-history \
      --use-train-pool \
      --save-eval-trajectories \
      --eval-trajectory-dir "${TRAJ_DIR}" 2>&1 | tee "${LOG_FILE}"
done

echo ""
echo "=========================================================================="
echo "L1_FC vs L0 eval done."
echo "  Per-seed dirs:    ${OUT_ROOT}/seed{N}/"
echo "  Trajectories:     ${OUT_ROOT}/seed{N}/trajectories/final/opponent_*.pkl"
echo "  Quick summary:    grep -E 'success_rate_per_opponent|All success' ${OUT_ROOT}/seed*/eval.log"
echo "=========================================================================="
