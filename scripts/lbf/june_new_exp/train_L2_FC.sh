#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# LBF L2 FC (Fully Compatible) training — horizon = 80.
#
# Trains L2 (player-id=0, LEFT) against the L1 FC h80 pool.
# L1 partners are RIGHT (player-id=1).
#
# L1 seed split:
#   Train (2 seeds): 1, 2   →  2 train partners
#   Eval  (1 seed):  3      →  1 eval partner
#
# Output: logs/LBF/ppo_june_new_exp/L2_FC_h80/seed{N}_seed{N}/
#         logs/LBF/june_new_exp/L2_FC_h80/training_seed{N}.log
#
# Usage:
#   bash scripts/lbf/june_new_exp/train_L2_FC.sh              # 3 seeds, GPUs 3,4,5
#   SEEDS="1 2" bash scripts/lbf/june_new_exp/train_L2_FC.sh
#   GPUS="0 1 2" SEEDS="1 2 3" bash scripts/lbf/june_new_exp/train_L2_FC.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

SEEDS=${SEEDS:-"1 2 3"}
GPUS=${GPUS:-"3 4 5"}
NUM_PROCS=${NUM_PROCS:-12}
read -ra SEED_ARR <<< "${SEEDS}"
read -ra GPU_ARR  <<< "${GPUS}"

TRAIN_SPEC='L1_FC_h80:1,2'
EVAL_SPEC='L1_FC_h80:3'
TRAIN_POOL_SIZE=2
EVAL_POOL_SIZE=1

mkdir -p logs/LBF/june_new_exp/L2_FC_h80

echo "=========================================================================="
echo "LBF L2 FC training (horizon=80)"
echo "  Seeds:      ${SEEDS}"
echo "  GPUs:       ${GPUS}"
echo "  Train spec: ${TRAIN_SPEC}  (${TRAIN_POOL_SIZE} partners)"
echo "  Eval spec:  ${EVAL_SPEC}   (${EVAL_POOL_SIZE} partner)"
echo "  Output:     logs/LBF/ppo_june_new_exp/L2_FC_h80/seed{N}_seed{N}/"
echo "=========================================================================="

launch_l2() {
    local seed=$1
    local gpu_id=$2
    local session="lbf_L2_FC_h80_seed${seed}"
    local log_file="logs/LBF/june_new_exp/L2_FC_h80/training_seed${seed}.log"

    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "Launching: LBF L2 FC h80 seed=${seed} GPU=${gpu_id} NUMA=${numa_node} (tmux: ${session})"
    tmux kill-session -t ${session} 2>/dev/null

    tmux new-session -d -s ${session} "
cd ${PACE_ROOT}
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
python train_.py --env-name LBF \
  --env_config './environment/lbf/config/lbf_fc_3type.yaml' \
  --lbf-num-fruit-types 3 \
  --lbf-horizon 80 \
  --horizon 80 \
  --player-id 0 \
  --algo ppo --num-steps 1000 --num-processes ${NUM_PROCS} --use-gae --eps 1e-8 --use-proper-time-limit \
  --lr 3e-4 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 \
  --max-grad-norm 15.0 \
  --num-env-steps 60000000 \
  --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
  --exp-name 'june_new_exp/L2_FC_h80/seed${seed}' \
  --latent-training --e2e-obj \
  --deterministic-latent \
  --use-latent-critic \
  --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
  --latent-dim 128 --kl-coef 0 \
  --hidden-dims 128 128 --act-func relu \
  --train-pool-size ${TRAIN_POOL_SIZE} --eval-pool-size ${EVAL_POOL_SIZE} \
  --rule-based-opponents 0 \
  --use-l1-specialized \
  --l1-specialized-train '${TRAIN_SPEC}' \
  --l1-specialized-eval  '${EVAL_SPEC}' \
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
  --p 0.0 \
  ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} 2>&1 | tee ${log_file}
"
}

n=${#SEED_ARR[@]}
for ((i=0; i<n; i++)); do
    launch_l2 "${SEED_ARR[$i]}" "${GPU_ARR[$i % ${#GPU_ARR[@]}]}"
    sleep 5
done

echo ""
echo "All ${n} LBF L2 FC h80 seeds launched."
echo "tmux sessions: $(printf 'lbf_L2_FC_h80_seed%s ' "${SEED_ARR[@]}")"
echo "Logs: logs/LBF/june_new_exp/L2_FC_h80/training_seed{N}.log"
echo "Checkpoints (after training): logs/LBF/ppo_june_new_exp/L2_FC_h80/seed{N}_seed{N}/ppo/"
