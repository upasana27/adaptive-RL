#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# L1 PC (specialist, knows 2 of 3 recipes) vs L0  — OR —  L2 PC vs L1 PC held-out.
#
# Same eval-time config (5-episode meta-round, --pop-oldest-episode sliding
# window, --self-obs-mode, --separate-history, --include-current-episode) so
# the two settings are directly comparable.
#
# MODE=L1_vs_L0 (default)
#   Diagnostic for the L2-vs-PC failure: characterise what each PC specialist L1
#   learned against L0 — does it LEAD or WAIT, does it commit to one fixed recipe,
#   and what does it do vs an out-of-set L0 (a recipe it cannot cook)?
#   Target pool: RIGHT PC L1s (player-id=1) that L2_PC actually trained against.
#   Each PC type × 3 L0 recipes (LO, TC, PB). Output opponent_{0,1,2}.
#
# MODE=L2_vs_L1
#   Re-runs the L2 eval with the L1-vs-L0 config (5-episode round instead of the
#   ad-hoc 20-episode one under logs/.../eval_L2_PC/). L2 is LEFT (player_id=0);
#   partners are RIGHT L1_PC held-out (LO_PB, LO_TC, PB_TC × seeds 8,9,10 = 9
#   partners). Output opponent_{0..8} using --partner-policy-paths ordering.
#
# Output roots:
#   MODE=L1_vs_L0 (SIDE=right): logs/.../eval_L1_PC_vs_L0/{TYPE}/seed{N}/
#   MODE=L1_vs_L0 (SIDE=left) : logs/.../eval_L1_PC_LEFT_vs_L0/{TYPE}/seed{N}/
#   MODE=L2_vs_L1             : logs/.../eval_L2_PC_vs_L1/seed{N}/
#
# Usage:
#   bash scripts/overcooked/june_new_exp/eval/right/eval_L1_PC_vs_L0.sh              # default: L1_vs_L0
#   MODE=L2_vs_L1 bash ...                                                            # L2 eval
#   MODE=L2_vs_L1 SEEDS="1" GPU=0 bash ...                                            # one L2 seed
#   MODE=L2_vs_L1 PARTNER_SEEDS="4 5 6" bash ...                                      # different held-out seeds
#
# Env vars honoured:
#   Common: TEST_EPISODES (default 5), GPU (default 0)
#   L1_vs_L0: TYPES, SEEDS, SIDE (left|right), L1_BASE_DIR
#   L2_vs_L1: SEEDS (L2 seeds), PARTNER_TYPES, PARTNER_SEEDS,
#             L2_BASE_DIR, L1_PARTNER_BASE, PARTNER_CKPT (default 30000000.pt)

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

MODE=${MODE:-L1_vs_L0}
GPU=${GPU:-0}
TEST_EPISODES=${TEST_EPISODES:-5}

# -------- Common eval flags. Match the L1_PC-vs-L0 setup exactly. --------
COMMON_FLAGS=(
  --env-name Overcooked
  --env_config './environment/overcooked/config/fc_small_test.yaml'
  --recipe-type cross
  --algo ppo
  --history-size 5
  --include-current-episode
  --merge-encoder-computation
  --use-meta-episode
  --self-obs-mode
  --separate-history
  --test-episodes "${TEST_EPISODES}"
  --p 0.0
  --save-eval-trajectories
)

# ============================================================================
if [[ "${MODE}" == "L1_vs_L0" ]]; then
# ============================================================================
    TYPES=${TYPES:-"LO_PB LO_TC PB_TC"}
    SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
    SIDE=${SIDE:-right}
    if [[ "${SIDE}" == "left" ]]; then
        PLAYER_ID=0
        L1_BASE_DIR=${L1_BASE_DIR:-${PACE_ROOT}/logs/Overcooked/june_new_exp/L1_PC/left}
        CKPT_SUBDIR='seed${SEED}'
        OUT_ROOT=logs/Overcooked/june_new_exp/eval_L1_PC_LEFT_vs_L0
    else
        PLAYER_ID=1
        L1_BASE_DIR=${L1_BASE_DIR:-${PACE_ROOT}/logs/Overcooked/ppo_L1_specialized}
        CKPT_SUBDIR='seed${SEED}_seed${SEED}'
        OUT_ROOT=logs/Overcooked/june_new_exp/eval_L1_PC_vs_L0
    fi
    mkdir -p "${OUT_ROOT}"

    echo "=========================================================================="
    echo "MODE=L1_vs_L0  (${SIDE})"
    echo "  Types:    ${TYPES}"
    echo "  Seeds:    ${SEEDS}"
    echo "  L1 base:  ${L1_BASE_DIR}"
    echo "  Episodes: ${TEST_EPISODES} per L0 partner"
    echo "  Out root: ${OUT_ROOT}"
    echo "=========================================================================="

    for TYPE in ${TYPES}; do
      for SEED in ${SEEDS}; do
        SEED_DIR=$(eval echo "${CKPT_SUBDIR}")
        L1_CKPT="${L1_BASE_DIR}/${TYPE}/${SEED_DIR}/ppo/30000000.pt"
        [ -f "${L1_CKPT}" ] || L1_CKPT="${L1_BASE_DIR}/${TYPE}/${SEED_DIR}/ppo/latest.pt"
        if [ ! -f "${L1_CKPT}" ]; then
            echo "SKIP ${TYPE} seed ${SEED}: no ckpt under ${L1_BASE_DIR}/${TYPE}/${SEED_DIR}/ppo/"
            continue
        fi

        OUT_DIR="${OUT_ROOT}/${TYPE}/seed${SEED}"
        LOG_DIR="${OUT_DIR}/log_internal"
        TRAJ_DIR="${OUT_DIR}/trajectories"
        LOG_FILE="${OUT_DIR}/eval.log"
        mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TRAJ_DIR}"

        echo ""
        echo "--- L1_PC ${TYPE} seed ${SEED}  vs 3 L0 partners  (ckpt: $(basename ${L1_CKPT})) ---"

        CUDA_VISIBLE_DEVICES=${GPU} python online_test_.py "${COMMON_FLAGS[@]}" \
          --exp-name "$(basename ${OUT_ROOT})/${TYPE}/seed${SEED}" \
          --log-dir "${LOG_DIR}" \
          --player-id ${PLAYER_ID} \
          --policy-dir "${L1_CKPT}" \
          --seed ${SEED} \
          --train-pool-size 3 \
          --eval-pool-size 0 \
          --rule-based-opponents 3 \
          --use-train-pool \
          --eval-trajectory-dir "${TRAJ_DIR}" 2>&1 | tee "${LOG_FILE}"
      done
    done

    echo ""
    echo "=========================================================================="
    echo "L1_PC vs L0 eval done. Trajectories: ${OUT_ROOT}/{TYPE}/seed{N}/trajectories/final/"
    echo "=========================================================================="

# ============================================================================
elif [[ "${MODE}" == "L2_vs_L1" ]]; then
# ============================================================================
    SEEDS=${SEEDS:-"1 2 3"}                             # L2 seeds
    PARTNER_TYPES=${PARTNER_TYPES:-"LO_PB LO_TC PB_TC"}
    PARTNER_SEEDS=${PARTNER_SEEDS:-"8 9 10"}             # L1 held-out seeds
    L2_BASE_DIR=${L2_BASE_DIR:-${PACE_ROOT}/logs/Overcooked/june_new_exp/L2_PC/left}
    L1_PARTNER_BASE=${L1_PARTNER_BASE:-${PACE_ROOT}/logs/Overcooked/ppo_L1_specialized}
    PARTNER_CKPT=${PARTNER_CKPT:-30000000.pt}
    PLAYER_ID=0                                          # L2 is LEFT
    OUT_ROOT=logs/Overcooked/june_new_exp/eval_L2_PC_vs_L1
    mkdir -p "${OUT_ROOT}"

    # Build --partner-policy-paths comma-separated list. Order determines opp_id.
    PARTNER_PATHS=""
    PARTNER_LABELS=""
    for T in ${PARTNER_TYPES}; do
      for PS in ${PARTNER_SEEDS}; do
        cand="${L1_PARTNER_BASE}/${T}/seed${PS}_seed${PS}/ppo/${PARTNER_CKPT}"
        [ -f "${cand}" ] || cand="${L1_PARTNER_BASE}/${T}/seed${PS}_seed${PS}/ppo/latest.pt"
        if [ ! -f "${cand}" ]; then
            echo "SKIP partner ${T} seed${PS}: no checkpoint under ${L1_PARTNER_BASE}/${T}/seed${PS}_seed${PS}/ppo/"
            continue
        fi
        if [ -z "${PARTNER_PATHS}" ]; then
          PARTNER_PATHS="${cand}"
          PARTNER_LABELS="${T}:seed${PS}"
        else
          PARTNER_PATHS="${PARTNER_PATHS},${cand}"
          PARTNER_LABELS="${PARTNER_LABELS} ${T}:seed${PS}"
        fi
      done
    done
    if [ -z "${PARTNER_PATHS}" ]; then
      echo "ERROR: no partner checkpoints found under ${L1_PARTNER_BASE}"
      exit 1
    fi

    echo "=========================================================================="
    echo "MODE=L2_vs_L1"
    echo "  L2 seeds:   ${SEEDS}"
    echo "  Partners:   ${PARTNER_LABELS}"
    echo "  Episodes:   ${TEST_EPISODES} per L1 partner  (== 1 meta-round)"
    echo "  L2 base:    ${L2_BASE_DIR}"
    echo "  Out root:   ${OUT_ROOT}"
    echo "=========================================================================="

    for SEED in ${SEEDS}; do
        L2_CKPT="${L2_BASE_DIR}/seed${SEED}/ppo/30000000.pt"
        [ -f "${L2_CKPT}" ] || L2_CKPT="${L2_BASE_DIR}/seed${SEED}/ppo/latest.pt"
        if [ ! -f "${L2_CKPT}" ]; then
            echo "SKIP L2 seed ${SEED}: no ckpt under ${L2_BASE_DIR}/seed${SEED}/ppo/"
            continue
        fi

        OUT_DIR="${OUT_ROOT}/seed${SEED}"
        LOG_DIR="${OUT_DIR}/log_internal"
        TRAJ_DIR="${OUT_DIR}/trajectories"
        LOG_FILE="${OUT_DIR}/eval.log"
        mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TRAJ_DIR}"

        # Record which partner opp_id corresponds to which (type, seed).
        {
          echo "# opp_id -> partner"
          i=0
          for lbl in ${PARTNER_LABELS}; do
            echo "opp_${i} ${lbl}"
            i=$((i+1))
          done
        } > "${OUT_DIR}/partner_manifest.txt"

        echo ""
        echo "--- L2 seed ${SEED}  vs $(echo ${PARTNER_LABELS} | wc -w) L1 partners  (ckpt: $(basename ${L2_CKPT})) ---"

        CUDA_VISIBLE_DEVICES=${GPU} python online_test_.py "${COMMON_FLAGS[@]}" \
          --exp-name "eval_L2_PC_vs_L1/seed${SEED}" \
          --log-dir "${LOG_DIR}" \
          --player-id ${PLAYER_ID} \
          --policy-dir "${L2_CKPT}" \
          --seed ${SEED} \
          --partner-policy-paths "${PARTNER_PATHS}" \
          --pop-oldest-episode \
          --eval-trajectory-dir "${TRAJ_DIR}" 2>&1 | tee "${LOG_FILE}"
    done

    echo ""
    echo "=========================================================================="
    echo "L2_PC vs L1 eval done. Trajectories: ${OUT_ROOT}/seed{N}/trajectories/final/"
    echo "Partner ordering saved per seed at: ${OUT_ROOT}/seed{N}/partner_manifest.txt"
    echo "=========================================================================="

else
    echo "ERROR: unknown MODE='${MODE}'. Use L1_vs_L0 or L2_vs_L1."
    exit 1
fi
