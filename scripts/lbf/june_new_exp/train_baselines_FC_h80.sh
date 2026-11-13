#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# LBF baseline training — FC domain, horizon=80.
#
# Trains 4 baselines against the L0 pool (3 fruit-type specialists).
# Baselines: LILI, LIAM, LIAMx, Generalist
# (PACE/NestRL-L1 is already trained — use L1 FC h80 checkpoints for that baseline.)
#
# All baselines train as player-id=0 (LEFT), 3 L0 opponents, 30M steps.
# 3 seeds each → 12 runs total; distributed across GPUs 6 and 7 (NUMA 1).
#
# Evaluation against held-out L1 FC h80 seed 3 is done separately.
#
# Output: logs/LBF/ppo_june_new_exp/{METHOD}_FC_h80/seed{N}_seed{N}/
#         logs/LBF/june_new_exp/{METHOD}_FC_h80/training_seed{N}.log
#
# Usage:
#   bash scripts/lbf/june_new_exp/train_baselines_FC_h80.sh          # all 4 baselines, seeds 1-3
#   SEEDS="1 2" GPUS="6 7" bash scripts/lbf/june_new_exp/train_baselines_FC_h80.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

SEEDS=${SEEDS:-"1 2 3"}
GPUS=${GPUS:-"6 7"}
L1_EVAL_SPEC=${L1_EVAL_SPEC:-"L1_FC_h80:3"}
read -ra SEED_ARR <<< "${SEEDS}"
read -ra GPU_ARR  <<< "${GPUS}"

mkdir -p logs/LBF/june_new_exp/LILI_FC_h80
mkdir -p logs/LBF/june_new_exp/LIAM_FC_h80
mkdir -p logs/LBF/june_new_exp/LIAMx_FC_h80
mkdir -p logs/LBF/june_new_exp/Generalist_FC_h80

echo "=========================================================================="
echo "LBF Baseline Training — FC domain, horizon=80"
echo "  Seeds:  ${SEEDS}"
echo "  GPUs:   ${GPUS} (NUMA 1)"
echo "  Methods: LILI, LIAM, LIAMx, Generalist"
echo "  Steps:  30M each"
echo "=========================================================================="

# GPU counter for round-robin assignment (incremented inline, not in subshell)
GPU_COUNTER=0
num_gpus=${#GPU_ARR[@]}

# Common LBF args shared by all baselines
common_args() {
    echo "--env-name LBF \
  --env_config './environment/lbf/config/lbf_fc_3type.yaml' \
  --lbf-num-fruit-types 3 \
  --lbf-horizon 80 \
  --horizon 80 \
  --player-id 0 \
  --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
  --lr 1e-3 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 \
  --max-grad-norm 15.0 \
  --num-env-steps 30000000 \
  --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
  --train-pool-size 3 --eval-pool-size 3 \
  --rule-based-opponents 3 \
  --lbf-l1-eval-spec '${L1_EVAL_SPEC}' \
  --history-size 5 \
  --p 0.0 \
  ${WANDB_USER:+--wandb-user-name ${WANDB_USER}}"
}

launch() {
    local method=$1
    local seed=$2
    local gpu_id=$3
    local extra_args="$4"
    local session="lbf_${method}_FC_h80_seed${seed}"
    local log_file="logs/LBF/june_new_exp/${method}_FC_h80/training_seed${seed}.log"
    local exp_name="june_new_exp/${method}_FC_h80/seed${seed}"

    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "  Launching ${method} seed=${seed} GPU=${gpu_id} NUMA=${numa_node} (tmux: ${session})"
    tmux kill-session -t ${session} 2>/dev/null

    tmux new-session -d -s ${session} "
cd ${PACE_ROOT}
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
python train_.py $(common_args) \
  --exp-name '${exp_name}' \
  --seed ${seed} \
  ${extra_args} 2>&1 | tee ${log_file}
"
}

# ── LILI ──────────────────────────────────────────────────────────────────────
# PACE/NestRL architecture + transition prediction auxiliary loss.
# Latent context trained e2e, but with explicit next-obs prediction head.
echo ""
echo "── LILI ──"
LILI_EXTRA="--latent-training --e2e-obj \
  --deterministic-latent \
  --use-latent-critic \
  --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
  --latent-dim 128 --kl-coef 0 \
  --hidden-dims 128 128 --act-func relu \
  --auxiliary-transition-pred-coef 1.0 --self-action-mode --last-episode-only --collect-next-obs \
  --history-middle-sampling \
  --merge-encoder-computation \
  --history-full-size 5 \
  --history-use-episodes \
  --include-current-episode \
  --use-meta-episode \
  --self-obs-mode \
  --clear-history-on-full \
  --separate-history"
for seed in "${SEED_ARR[@]}"; do
    launch "LILI" "${seed}" "${GPU_ARR[$((GPU_COUNTER % num_gpus))]}" "${LILI_EXTRA}"
    GPU_COUNTER=$((GPU_COUNTER + 1))
    sleep 3
done

# ── LIAM ──────────────────────────────────────────────────────────────────────
# GRU + joint training + peer obs/act prediction. No meta-episode.
echo ""
echo "── LIAM ──"
LIAM_EXTRA="--joint-training \
  --hidden-dims 128 128 --act-func relu \
  --recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic \
  --all-has-last-action \
  --collect-peer-traj \
  --auxiliary-peer-obs-pred-coef 1.0 \
  --auxiliary-peer-act-pred-coef 1.0"
for seed in "${SEED_ARR[@]}"; do
    launch "LIAM" "${seed}" "${GPU_ARR[$((GPU_COUNTER % num_gpus))]}" "${LIAM_EXTRA}"
    GPU_COUNTER=$((GPU_COUNTER + 1))
    sleep 3
done

# ── LIAMx ─────────────────────────────────────────────────────────────────────
# LIAM + meta-episode (gives partner-type context across episodes).
echo ""
echo "── LIAMx ──"
LIAMx_EXTRA="--joint-training \
  --hidden-dims 128 128 --act-func relu \
  --recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic \
  --all-has-last-action \
  --collect-peer-traj \
  --auxiliary-peer-obs-pred-coef 1.0 \
  --auxiliary-peer-act-pred-coef 1.0 \
  --use-meta-episode"
for seed in "${SEED_ARR[@]}"; do
    launch "LIAMx" "${seed}" "${GPU_ARR[$((GPU_COUNTER % num_gpus))]}" "${LIAMx_EXTRA}"
    GPU_COUNTER=$((GPU_COUNTER + 1))
    sleep 3
done

# ── Generalist ─────────────────────────────────────────────────────────────────
# GRU + joint training + meta-episode. No peer prediction auxiliary loss.
echo ""
echo "── Generalist ──"
GEN_EXTRA="--joint-training \
  --hidden-dims 128 128 --act-func relu \
  --recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic \
  --use-meta-episode"
for seed in "${SEED_ARR[@]}"; do
    launch "Generalist" "${seed}" "${GPU_ARR[$((GPU_COUNTER % num_gpus))]}" "${GEN_EXTRA}"
    GPU_COUNTER=$((GPU_COUNTER + 1))
    sleep 3
done

echo ""
echo "=========================================================================="
echo "All 12 baseline runs launched."
echo "tmux sessions:"
for method in LILI LIAM LIAMx Generalist; do
    printf "  %s: " "${method}"
    for seed in "${SEED_ARR[@]}"; do printf "lbf_%s_FC_h80_seed%s " "${method}" "${seed}"; done
    echo ""
done
echo ""
echo "Logs: logs/LBF/june_new_exp/{METHOD}_FC_h80/training_seed{N}.log"
echo "Checkpoints: logs/LBF/ppo_june_new_exp/{METHOD}_FC_h80/seed{N}_seed{N}/ppo/"
echo "=========================================================================="
