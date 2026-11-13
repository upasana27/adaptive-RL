#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Train L1 LEFT PC agents (PC partners) — each L1 knows 2 of 3 recipes.
#
# Mirror of the RIGHT-side PC pool we trained earlier, now on player-id=0.
# Three pair-types × N seeds each.
#
# Pool seed mappings (from L0 generation in policy.py):
#   pool_seed=1, pool_size=2 → ['LO', 'TC']
#   pool_seed=2, pool_size=2 → ['TC', 'PB']
#   pool_seed=3, pool_size=2 → ['PB', 'LO']
# So agent types:
#   LO_PB → pool_seed=3 (knows LO + PB)
#   LO_TC → pool_seed=1 (knows LO + TC)
#   PB_TC → pool_seed=2 (knows PB + TC)
#
# Output: logs/Overcooked/ppo_june_new_exp/L1_LEFT_PC/{TYPE}/seed{N}_seed{N}/ppo/
#
# Usage:
#   bash scripts/overcooked/june_new_exp/train/left/train_L1_PC.sh                 # default 3 types × 10 seeds
#   SEEDS="1 2 3" TYPES="LO_PB" bash ...                                                   # subset
#   GPUS="0 1 2 3 7" SEEDS="1 2 3 4 5 6 7 8 9 10" bash ...                                  # explicit GPU set

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
TYPES=${TYPES:-"LO_PB LO_TC PB_TC"}
GPUS=${GPUS:-"0 1 2 3 7"}
# L0 action-noise probability used during L1 training. 0.0 keeps the L0
# rule-based partner fully deterministic (original PC). Larger P makes the
# L0 pool noisier, to probe whether resulting L1s are more robust.
P=${P:-0.0}
# Which side is being trained. Defaults to left for backward compat.
SIDE=${SIDE:-left}
# Peer-identifier (auxiliary policy classifier).
# PEER_ID_COEF: CE loss weight on the classifier head. Empty = off (no head).
# PEER_ID_REWARD: reward coefficient shaping the actor by classifier accuracy.
#                 Empty = off (no reward). Setting this without PEER_ID_COEF
#                 also enables the head so the reward has a trained classifier.
PEER_ID_COEF=${PEER_ID_COEF:-}
PEER_ID_REWARD=${PEER_ID_REWARD:-}
PEER_ID_TYPE=${PEER_ID_TYPE:-accuracy}     # accuracy | entropy
PEER_ID_MODE=${PEER_ID_MODE:-diff}         # diff | full | max_diff | max_full

read -ra SEED_ARR <<< "${SEEDS}"
read -ra TYPE_ARR <<< "${TYPES}"
read -ra GPU_ARR  <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}

# Isolate output dirs per P so different noise settings don't collide.
if [[ "${P}" == "0.0" || "${P}" == "0" ]]; then
    RUN_TAG=""
else
    # Convert P to a safe filesystem tag, e.g. 0.25 -> p025
    RUN_TAG="_p$(echo ${P} | tr -d '.')"
fi

# Peer-identifier tag appended to output paths (so it doesn't collide with a
# vanilla same-P run).
PEER_TAG=""
if [[ -n "${PEER_ID_COEF}" || -n "${PEER_ID_REWARD}" ]]; then
    PEER_TAG="_peerid"
fi

# player_id from SIDE
if [[ "${SIDE}" == "right" ]]; then PLAYER_ID=1
elif [[ "${SIDE}" == "left" ]]; then PLAYER_ID=0
else echo "ERROR: SIDE must be left or right, got '${SIDE}'"; exit 1
fi

FULL_TAG="${RUN_TAG}${PEER_TAG}"

# pool_seed by type
declare -A POOL_SEED
POOL_SEED[LO_PB]=3
POOL_SEED[LO_TC]=1
POOL_SEED[PB_TC]=2

LOG_ROOT=logs/Overcooked/june_new_exp/L1_PC${FULL_TAG}/${SIDE}
mkdir -p "${LOG_ROOT}"

echo "=========================================================================="
echo "L1 ${SIDE^^} PC (PC partners)"
echo "  Types:  ${TYPES}"
echo "  Seeds:  ${SEEDS}"
echo "  GPUs:   ${GPUS}"
echo "  P (L0 noise):   ${P}"
echo "  Peer-ID coef:   ${PEER_ID_COEF:-<off>}"
echo "  Peer-ID reward: ${PEER_ID_REWARD:-<off>}  (type=${PEER_ID_TYPE}, mode=${PEER_ID_MODE})"
echo "  Output tag:     L1_PC${FULL_TAG}/${SIDE}"
echo "  Total:  $(( ${#TYPE_ARR[@]} * ${#SEED_ARR[@]} )) runs (30M env steps each)"
echo "=========================================================================="

launch() {
    local seed=$1; local agent_type=$2; local gpu_id=$3
    local pool_seed=${POOL_SEED[${agent_type}]}
    local exp_dir="june_new_exp/L1_PC${FULL_TAG}/${SIDE}/${agent_type}/seed${seed}"
    local log_file="${LOG_ROOT}/${agent_type}_seed${seed}.log"

    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "Launching: ${agent_type}_seed${seed} (GPU=${gpu_id} NUMA=${numa_node}, pool_seed=${pool_seed})"

    local peer_flags=""
    if [[ -n "${PEER_ID_COEF}" ]]; then
        peer_flags="${peer_flags} --auxiliary-policy-cls-coef ${PEER_ID_COEF}"
    fi
    if [[ -n "${PEER_ID_REWARD}" ]]; then
        # Enable classifier head via aux coef if the user only asked for reward
        [[ -z "${PEER_ID_COEF}" ]] && peer_flags="${peer_flags} --auxiliary-policy-cls-coef 0.0"
        peer_flags="${peer_flags} --policy-cls-reward-coef ${PEER_ID_REWARD}"
        peer_flags="${peer_flags} --policy-cls-reward-type ${PEER_ID_TYPE}"
        peer_flags="${peer_flags} --policy-cls-reward-mode ${PEER_ID_MODE}"
    fi

    ( CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config './environment/overcooked/config/fc_small_test.yaml' \
      --player-id ${PLAYER_ID} --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr 1e-3 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps 30000000 \
      --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
      --exp-name "${exp_dir}" \
      --latent-training --e2e-obj \
      --deterministic-latent \
      --use-latent-critic \
      --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
      --latent-dim 128 --kl-coef 0 \
      --hidden-dims 128 128 --act-func relu \
      --train-pool-size 2 --eval-pool-size 2 \
      --rule-based-opponents 2 \
      --pool-seed ${pool_seed} \
      --history-middle-sampling \
      --merge-encoder-computation \
      --history-full-size 5 \
      --history-size 5 \
      --history-use-episodes \
      --include-current-episode \
      --use-meta-episode \
      --self-obs-mode \
      --clear-history-on-full \
      --separate-history \
      --seed ${seed} \
      --p ${P} \
      ${peer_flags} \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > ${log_file} 2>&1 && \
    bash ${PACE_ROOT}/scripts/overcooked/june_new_exp/consolidate_checkpoints.sh "L1_PC${FULL_TAG}" "${SIDE}" >> ${log_file} 2>&1 ) &
}

gpu_idx=0
for agent_type in "${TYPE_ARR[@]}"; do
    echo ""
    echo "--- Type: ${agent_type} ---"
    for seed in "${SEED_ARR[@]}"; do
        gpu=${GPU_ARR[$((gpu_idx % NUM_GPUS))]}
        launch ${seed} ${agent_type} ${gpu}
        gpu_idx=$((gpu_idx + 1))
        sleep 2
    done
done

echo ""
echo "All L1 LEFT PC jobs launched."
echo "Monitor: tail -f ${LOG_ROOT}/*.log"
