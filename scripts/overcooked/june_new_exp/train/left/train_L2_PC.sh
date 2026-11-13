#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# L2 partial-overlap (Exp 1) — June rerun under the june_new_exp folder.
#
# Trains L2 LEFT (player-id=0) against the partially-compatible specialist
# L1 RIGHT pool (LO_PB, LO_TC, PB_TC — each L1 knows 2 of 3 recipes).
#
# Three-tier split of the 10 seeds per type:
#   Train (5 per type):   seeds 1,2,3,4,5    -> 15 train partners
#   Eval-during-training (2 per type): 6,7   -> 6 eval partners
#   Final held-out test (3 per type): 8,9,10 -> 9 partners, NOT seen during training
#
# Output: logs/Overcooked/ppo_june_new_exp/L2_PC/seed{N}_seed{N}/
# (the leading 'ppo_' is automatically prepended by train_.py.)
#
# GPUs 4, 5, 6 (NUMA node 1), one seed each, NUMA-pinned.
#
# Usage:
#   bash scripts/overcooked/june_new_exp/train_L2_PC.sh           # all 3 seeds
#   SEEDS="1 2" bash scripts/overcooked/june_new_exp/train_L2_PC.sh
#   GPUS="4 5 6" SEEDS="1 2 3" bash scripts/overcooked/june_new_exp/train_L2_PC.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

SEEDS=${SEEDS:-"1 2 3"}
GPUS=${GPUS:-"4 5 6"}
read -ra SEED_ARR <<< "${SEEDS}"
read -ra GPU_ARR  <<< "${GPUS}"

# --- Overridable via env vars. Defaults reproduce the original non-peer-id L2_PC
#     (ppo_L1_specialized pool, 5 train + 2 eval per type). Override to train the
#     peer-id variant against the peer-id L1 pool with a 2-train/1-eval split, e.g.:
#       EXP_TAG=L2_PC_peerid \
#       L1_BASE_DIR=.../logs/Overcooked/june_new_exp/L1_PC_p05_peerid/right \
#       TRAIN_SPEC='LO_PB:1,2;LO_TC:1,2;PB_TC:1,2' EVAL_SPEC='LO_PB:3;LO_TC:3;PB_TC:3' \
#       TRAIN_POOL_SIZE=6 EVAL_POOL_SIZE=3 \
#       EXTRA_ARGS='--auxiliary-policy-cls-coef 1.0 --policy-cls-reward-coef 0.1 --policy-cls-reward-type accuracy --policy-cls-reward-mode diff' \
#       SEEDS='1 2 3' GPUS='1 2 3' bash scripts/overcooked/june_new_exp/train/left/train_L2_PC.sh
EXP_TAG=${EXP_TAG:-L2_PC}
# ENV_CONFIG override: train on a horizon variant (fc_small_test_h80.yaml).
# Default = the canonical horizon-40 config, so existing invocations are unchanged.
# NOTE: --horizon-override is a DEAD FLAG in this repo; horizon MUST come from the config.
ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml}
TRAIN_SPEC=${TRAIN_SPEC:-'LO_PB:1,2,3,4,5;LO_TC:1,2,3,4,5;PB_TC:1,2,3,4,5'}
EVAL_SPEC=${EVAL_SPEC:-'LO_PB:6,7;LO_TC:6,7;PB_TC:6,7'}
TRAIN_POOL_SIZE=${TRAIN_POOL_SIZE:-15}
EVAL_POOL_SIZE=${EVAL_POOL_SIZE:-6}
L1_BASE_DIR=${L1_BASE_DIR:-}          # empty -> default ppo_L1_specialized in policy.py
EXTRA_ARGS=${EXTRA_ARGS:-}            # extra train_.py flags (e.g. peer-id head/reward)
export WANDB_MODE=${WANDB_MODE:-offline}

mkdir -p logs/Overcooked/june_new_exp/${EXP_TAG}

echo "=========================================================================="
echo "Exp 1 — L2 partial-overlap (june_new_exp)"
echo "  Seeds: ${SEEDS}"
echo "  GPUs:  ${GPUS} (NUMA node 1)"
echo "  Train:  ${TRAIN_SPEC} (15 partners)"
echo "  Eval:   ${EVAL_SPEC}  (6 partners)"
echo "  Held-out final test (NOT used in training): LO_PB:8,9,10;LO_TC:8,9,10;PB_TC:8,9,10"
echo "=========================================================================="

launch_l2() {
    local seed=$1
    local gpu_id=$2
    local session="l2_${EXP_TAG}_seed${seed}"
    local log_file="logs/Overcooked/june_new_exp/${EXP_TAG}/training_seed${seed}.log"

    # GPUs 4-7 are on NUMA node 1 (CPUs 28-55, 84-111).
    # GPUs 0-3 are on NUMA node 0 (CPUs 0-27, 56-83).
    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "Launching: L2 partial-overlap seed=${seed} GPU=${gpu_id} NUMA=${numa_node} (tmux: ${session})"
    tmux kill-session -t ${session} 2>/dev/null

    tmux new-session -d -s ${session} "
cd ${PACE_ROOT}
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
python train_.py --env-name Overcooked \
  --env_config "${ENV_CONFIG}" \
  --player-id 0 --recipe-type cross \
  --algo ppo --num-steps 1000 --num-processes 15 --use-gae --eps 1e-8 --use-proper-time-limit \
  --lr 3e-4 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
  --max-grad-norm 15.0 \
  --num-env-steps 60000000 \
  --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
  --exp-name 'june_new_exp/${EXP_TAG}/left/seed${seed}' \
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
  ${L1_BASE_DIR:+--l1-specialized-base-dir ${L1_BASE_DIR}} \
  ${EXTRA_ARGS} \
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

echo 'Training done. Consolidating checkpoints...'
bash ${PACE_ROOT}/scripts/overcooked/june_new_exp/consolidate_checkpoints.sh ${EXP_TAG} left
"
}

n=${#SEED_ARR[@]}
for ((i=0; i<n; i++)); do
    launch_l2 "${SEED_ARR[$i]}" "${GPU_ARR[$i % ${#GPU_ARR[@]}]}"
    sleep 5
done

echo ""
echo "All ${n} L2 partial-overlap seeds launched."
echo "tmux sessions: $(printf 'l2_PC_seed%s ' "${SEED_ARR[@]}")"
echo "Logs: logs/Overcooked/june_new_exp/L2_PC/training_seed{N}.log"
echo "Checkpoints (after training): logs/Overcooked/ppo_june_new_exp/L2_PC/seed{N}_seed{N}/ppo/"
