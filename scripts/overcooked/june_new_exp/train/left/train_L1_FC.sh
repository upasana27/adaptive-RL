#!/bin/bash
# Train L1 LEFT FC (FC-type) agents — multi-recipe L1s trained against all 3 L0s.
#
# Mirrors the existing L1_left_p025 pool (p=0.25 stochastic L0 partners). Outputs go into
# the same checkpoint layout the existing pool uses, so downstream loaders work unchanged:
#   logs/Overcooked/L1_agents/L1_left_p025/seed{N}/ppo/30000000.pt
#
# (NOT inside the june_new_exp/ tree because we want these to slot into the existing pool
#  that is referenced elsewhere. Only the train script lives in june_new_exp/.)
#
# 30M env steps per seed, NUMA-pinned for GPUs 0-3 (node 0) / 4-7 (node 1).
#
# Usage:
#   bash scripts/overcooked/june_new_exp/train/left/train_L1_FC.sh                # SEEDS="9 10"
#   SEEDS="9 10 11" GPUS="0 1 2" bash scripts/overcooked/june_new_exp/train/left/train_L1_FC.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

SEEDS=${SEEDS:-"9 10"}
GPUS=${GPUS:-"0 3"}
read -ra SEED_ARR <<< "${SEEDS}"
read -ra GPU_ARR  <<< "${GPUS}"

OUT_ROOT=logs/Overcooked/L1_agents/L1_left_p025      # slots into existing pool
LOG_ROOT=logs/Overcooked/june_new_exp/L1_LEFT_FC
mkdir -p "${LOG_ROOT}"

echo "=========================================================================="
echo "Training L1 LEFT FC (p=0.25 L0s) seeds: ${SEEDS}"
echo "  Output checkpoints: ${OUT_ROOT}/seed{N}/ppo/"
echo "  Stdout logs:        ${LOG_ROOT}/seed{N}.log"
echo "  GPUs: ${GPUS}, NUMA-pinned by GPU"
echo "=========================================================================="

launch_seed() {
    local seed=$1
    local gpu_id=$2
    local log_file="${LOG_ROOT}/seed${seed}.log"

    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "Launching: L1 LEFT FC seed=${seed} GPU=${gpu_id} NUMA=${numa_node}"

    CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config './environment/overcooked/config/fc_small_test.yaml' \
      --player-id 0 --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr 3e-4 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps 30000000 \
      --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
      --exp-name "L1_agents/L1_left_p025/seed${seed}" \
      --latent-training --e2e-obj \
      --deterministic-latent \
      --use-latent-critic \
      --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
      --latent-dim 128 --kl-coef 0 \
      --hidden-dims 128 128 --act-func relu \
      --train-pool-size 3 --eval-pool-size 3 \
      --rule-based-opponents 3 \
      --pool-seed 1 \
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
      --p 0.25 \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > ${log_file} 2>&1 &
    echo "  PID=$! → ${log_file}"
}

n=${#SEED_ARR[@]}
for ((i=0; i<n; i++)); do
    launch_seed "${SEED_ARR[$i]}" "${GPU_ARR[$i % ${#GPU_ARR[@]}]}"
    sleep 3
done

echo ""
echo "Launched ${n} L1 LEFT FC training jobs in background."
echo "Monitor: tail -f ${LOG_ROOT}/seed*.log"
