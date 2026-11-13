#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Train RIGHT-side L1_PC_p05_peerid pool — extends the existing seed1 to a full
# 8-seed pool. rand_p=0.5 L0 (PC specialized pool is already 0,0,p_max, so --p
# feeds rand_p only) + peer-identifier head/reward.
#
# Differs from the frozen train_L1_PC.sh only by:
#   * CONCURRENCY CAP (MAXJOBS) — train_L1_PC.sh launches every seed at once,
#     which is what swamped the box before. Here at most MAXJOBS run at a time.
#   * WANDB_MODE=offline — online wandb deadlocks the loop on 403s.
#   * default SEEDS "2 3 4 5 6 7 8" — seed1 already exists on the right.
#
# Output (matches existing pool): june_new_exp/L1_PC_p05_peerid/right/{TYPE}/seed{N}
#
# Usage (durable):
#   setsid nohup bash scripts/overcooked/june_new_exp/train/right/train_L1_PC_p05_peerid_batched.sh \
#       > logs/Overcooked/june_new_exp/L1_PC_p05_peerid/right/_launcher.log 2>&1 < /dev/null &

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline
set -uo pipefail

REPO=${PACE_ROOT}
cd "${REPO}"

SIDE=right
PLAYER_ID=1
P=${P:-0.5}
PEER_ID_COEF=${PEER_ID_COEF:-1.0}
PEER_ID_REWARD=${PEER_ID_REWARD:-0.1}
PEER_ID_TYPE=${PEER_ID_TYPE:-accuracy}
PEER_ID_MODE=${PEER_ID_MODE:-diff}
SEEDS=${SEEDS:-"2 3 4 5 6 7 8"}
TYPES=${TYPES:-"LO_PB LO_TC PB_TC"}
GPUS=${GPUS:-"1 2 3"}
MAXJOBS=${MAXJOBS:-3}

# P -> filesystem tag (0.5 -> _p05), + _peerid, matching train_L1_PC.sh.
RUN_TAG="_p$(echo ${P} | tr -d '.')"
FULL_TAG="${RUN_TAG}_peerid"

read -ra SEED_ARR <<< "${SEEDS}"
read -ra TYPE_ARR <<< "${TYPES}"
read -ra GPU_ARR  <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}

declare -A POOL_SEED
POOL_SEED[LO_PB]=3
POOL_SEED[LO_TC]=1
POOL_SEED[PB_TC]=2

LOG_ROOT=logs/Overcooked/june_new_exp/L1_PC${FULL_TAG}/${SIDE}
mkdir -p "${LOG_ROOT}"

echo "=========================================================================="
echo "L1 RIGHT PC ${FULL_TAG}  (rand_p=${P} + peer-id, batched)"
echo "  Peer-ID: coef=${PEER_ID_COEF} reward=${PEER_ID_REWARD} (type=${PEER_ID_TYPE} mode=${PEER_ID_MODE})"
echo "  Seeds:   ${SEEDS}   Types: ${TYPES}"
echo "  GPUs:    ${GPUS}   MAXJOBS=${MAXJOBS}"
echo "  Output:  ${LOG_ROOT}"
echo "  Total:   $(( ${#TYPE_ARR[@]} * ${#SEED_ARR[@]} )) runs (30M steps each)"
echo "  Started: $(date '+%F %T')"
echo "=========================================================================="

run_one() {
    local seed=$1 agent_type=$2 gpu_id=$3
    local pool_seed=${POOL_SEED[${agent_type}]}
    local exp_dir="june_new_exp/L1_PC${FULL_TAG}/${SIDE}/${agent_type}/seed${seed}"
    local log_file="${LOG_ROOT}/${agent_type}_seed${seed}.log"
    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "[$(date '+%T')] START ${agent_type}_seed${seed} (GPU=${gpu_id} NUMA=${numa_node} pool_seed=${pool_seed})"
    CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
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
      --auxiliary-policy-cls-coef ${PEER_ID_COEF} \
      --policy-cls-reward-coef ${PEER_ID_REWARD} \
      --policy-cls-reward-type ${PEER_ID_TYPE} \
      --policy-cls-reward-mode ${PEER_ID_MODE} \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log_file}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        bash "${REPO}/scripts/overcooked/june_new_exp/consolidate_checkpoints.sh" "L1_PC${FULL_TAG}" "${SIDE}" >> "${log_file}" 2>&1
        echo "[$(date '+%T')] DONE  ${agent_type}_seed${seed} (rc=0)"
    else
        echo "[$(date '+%T')] FAIL  ${agent_type}_seed${seed} (rc=${rc}) — see ${log_file}"
    fi
}

# Seed-major order so all 3 types make progress together (balanced pool sooner),
# instead of finishing all LO_PB seeds before touching LO_TC / PB_TC.
JOBS=()
for seed in "${SEED_ARR[@]}"; do
    for agent_type in "${TYPE_ARR[@]}"; do
        JOBS+=("${agent_type} ${seed}")
    done
done

gpu_idx=0
active=0
for job in "${JOBS[@]}"; do
    read -r jtype jseed <<< "${job}"
    gpu=${GPU_ARR[$((gpu_idx % NUM_GPUS))]}
    run_one "${jseed}" "${jtype}" "${gpu}" &
    gpu_idx=$((gpu_idx + 1))
    active=$((active + 1))
    if [[ ${active} -ge ${MAXJOBS} ]]; then
        wait -n
        active=$((active - 1))
    fi
    sleep 3
done
wait

echo "=========================================================================="
echo "L1 RIGHT PC ${FULL_TAG} — all runs complete.  $(date '+%F %T')"
echo "=========================================================================="
