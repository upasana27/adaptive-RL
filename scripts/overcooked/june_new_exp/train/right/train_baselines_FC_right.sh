#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Train RIGHT-side baseline L1 agents (PACE, LiAM, LiAMx, LiLi, Generalist) in the
# FC domain (fc_small_test), player-id 1, against the 3-L0 FC pool.
#
# Purpose: give us baseline agents on the RIGHT so we can evaluate the LEFT-side
# FC L2 agents (pure FC = L2_left_big, and Loft_FC) paired with a baseline partner
# on the other side.
#
# Regime is mirrored from the FC L1 RIGHT training (config.yaml of
# june_new_exp/L1_FC/right): fc_small_test env, --player-id 1, --recipe-type cross,
# 3 rule-based L0 opponents, --pool-seed 1, --p 0.25, 12 procs, 18 mini-batch,
# 30M steps, --eval-episodes 5. Each baseline's *architecture* flags are copied
# verbatim from the frozen scripts/overcooked/{pace,liam,liamx,lili,generalist}.sh
# (those frozen scripts are NOT modified). ONE deviation: the frozen PACE/LiLi
# scripts pass `--visit-reward-coef 0.0 --visit-reward-type episode` (visit reward
# OFF, coef 0.0). That stats wrapper asserts player_id==0 (learning/envs.py:97), so
# it is dropped here for the right side. coef=0.0 means this changes nothing about
# training — only the incompatible player-0 stats collection is removed.
#
# Output (stdout):     logs/Overcooked/june_new_exp/baselines_FC/right/{METHOD}_seed{N}.log
# Output (checkpoints): logs/Overcooked/june_new_exp/baselines_FC/right/{METHOD}/seed{N}/ppo/
#                       (after consolidate_checkpoints.sh strips the _seed{N} suffix)
#
# Usage (durable, full 30M run):
#   mkdir -p logs/Overcooked/june_new_exp/baselines_FC/right
#   setsid nohup bash scripts/overcooked/june_new_exp/train/right/train_baselines_FC_right.sh \
#       > logs/Overcooked/june_new_exp/baselines_FC/right/_launcher.log 2>&1 < /dev/null &
#
# Smoke test (fast, throwaway group):
#   NUM_ENV_STEPS=60000 SAVE_INTERVAL=48000 EVAL_INTERVAL=24000 \
#   EXP_GROUP=baselines_FC_smoke MAXJOBS=5 \
#   bash scripts/overcooked/june_new_exp/train/right/train_baselines_FC_right.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline
set -uo pipefail

REPO=${PACE_ROOT}
cd "${REPO}"

# SIDE/PLAYER_ID overridable so the same script can train LEFT-side L1 baselines
# (SIDE=left PLAYER_ID=0) for the L1-vs-heldout-L1 comparison. Defaults unchanged.
SIDE=${SIDE:-right}
PLAYER_ID=${PLAYER_ID:-1}
# BOTH_SIDES=1 (2026-10-21): one policy that plays BOTH seats via --both-sides (even env procs
# LEFT, odd RIGHT -- see train_L1_bothsides.sh). Forces SIDE=bothsides, PLAYER_ID=0. Default off.
BOTH_SIDES=${BOTH_SIDES:-0}
BOTH_SIDES_FLAG=""
if [[ "${BOTH_SIDES}" == "1" ]]; then SIDE=bothsides; PLAYER_ID=0; BOTH_SIDES_FLAG="--both-sides"; fi
# torch opens one BLAS thread per core in every env worker otherwise (measured 2026-10-20)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=${P:-0.25}
POOL_SEED=${POOL_SEED:-1}
SEEDS=${SEEDS:-"1"}
METHODS=${METHODS:-"PACE LiAM LiAMx LiLi Generalist"}
GPUS=${GPUS:-"5 6 7 0"}
MAXJOBS=${MAXJOBS:-5}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-30000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
EXP_GROUP=${EXP_GROUP:-baselines_FC}

LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_GROUP}/${SIDE}
mkdir -p "${LOG_ROOT}"

read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}

echo "=========================================================================="
echo "RIGHT-side FC baselines  (group=${EXP_GROUP})"
echo "  Methods: ${METHODS}"
echo "  Seeds:   ${SEEDS}    GPUs: ${GPUS}   MAXJOBS=${MAXJOBS}"
echo "  Regime:  fc_small_test, player-id ${PLAYER_ID}, 3xL0, pool-seed ${POOL_SEED}, --p ${P}"
echo "  Steps:   ${NUM_ENV_STEPS}  (save@${SAVE_INTERVAL} eval@${EVAL_INTERVAL})"
echo "  Output:  ${LOG_ROOT}"
echo "  Started: $(date '+%F %T')"
echo "=========================================================================="

# Per-method architecture flags, copied verbatim from the frozen baseline scripts.
method_flags() {
    case "$1" in
        PACE)
            echo "--latent-training --e2e-obj --deterministic-latent --use-latent-critic \
--pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
--latent-dim 128 --kl-coef 0 --hidden-dims 128 128 --act-func relu \
--auxiliary-policy-cls-coef 1.0 --policy-cls-reward-coef 0.2 --policy-cls-reward-type accuracy \
--policy-cls-reward-mode max_full --policy-cls-warmup-steps 1000000 --policy-cls-reward-decay-steps 25000000 \
--history-middle-sampling --merge-encoder-computation \
--history-full-size 5 --history-size 5 --history-use-episodes --include-current-episode --use-meta-episode \
--self-obs-mode --clear-history-on-full --separate-history" ;;
        LiAM)
            echo "--joint-training --hidden-dims 128 128 --act-func relu \
--recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic \
--all-has-last-action --collect-peer-traj \
--auxiliary-peer-obs-pred-coef 1.0 --auxiliary-peer-act-pred-coef 1.0 --history-size 5" ;;
        LiAMx)
            echo "--joint-training --hidden-dims 128 128 --act-func relu \
--recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic \
--all-has-last-action --collect-peer-traj \
--auxiliary-peer-obs-pred-coef 1.0 --auxiliary-peer-act-pred-coef 1.0 --use-meta-episode --history-size 5" ;;
        LiLi)
            echo "--latent-training --e2e-obj --deterministic-latent --use-latent-critic \
--pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
--latent-dim 128 --kl-coef 0 --hidden-dims 128 128 --act-func relu \
--auxiliary-transition-pred-coef 1.0 --self-action-mode --last-episode-only --collect-next-obs \
--history-middle-sampling --merge-encoder-computation \
--history-full-size 5 --history-size 5 --history-use-episodes --include-current-episode --use-meta-episode \
--self-obs-mode --clear-history-on-full --separate-history" ;;
        Generalist)
            echo "--joint-training --hidden-dims 128 128 --act-func relu \
--recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic \
--use-meta-episode --history-size 5" ;;
        *)
            echo "UNKNOWN_METHOD_$1" ;;
    esac
}

run_one() {
    local method=$1 seed=$2 gpu=$3
    local exp_dir="june_new_exp/${EXP_GROUP}/${SIDE}/${method}/seed${seed}"
    local log_file="${LOG_ROOT}/${method}_seed${seed}.log"
    local numa_node
    if [[ "${gpu}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "[$(date '+%T')] START ${method}_seed${seed} (GPU=${gpu} NUMA=${numa_node})"
    CUDA_VISIBLE_DEVICES=${gpu} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config ./environment/overcooked/config/fc_small_test.yaml \
      --player-id ${PLAYER_ID} ${BOTH_SIDES_FLAG} --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr 1e-3 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps ${NUM_ENV_STEPS} \
      --save-interval ${SAVE_INTERVAL} --log-interval 10000 --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
      --exp-name "${exp_dir}" \
      --train-pool-size 3 --eval-pool-size 3 \
      --rule-based-opponents 3 \
      --pool-seed ${POOL_SEED} \
      --seed ${seed} \
      --p ${P} \
      $(method_flags "${method}") \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log_file}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "[$(date '+%T')] DONE  ${method}_seed${seed} (rc=0)"
    else
        echo "[$(date '+%T')] FAIL  ${method}_seed${seed} (rc=${rc}) — see ${log_file}"
    fi
}
# NOTE: checkpoint consolidation is done ONCE at the very end (after `wait`), not
# per-run. consolidate_checkpoints.sh globs every seed*_seed* dir under the group,
# so calling it while sibling methods are still training would move their live
# checkpoint dirs out from under them (races -> wandb FileNotFoundError / lost ckpts).

JOBS=()
for seed in ${SEEDS}; do
    for method in ${METHODS}; do
        JOBS+=("${method} ${seed}")
    done
done

gpu_idx=0
active=0
for job in "${JOBS[@]}"; do
    read -r jmethod jseed <<< "${job}"
    gpu=${GPU_ARR[$((gpu_idx % NUM_GPUS))]}
    run_one "${jmethod}" "${jseed}" "${gpu}" &
    gpu_idx=$((gpu_idx + 1))
    active=$((active + 1))
    if [[ ${active} -ge ${MAXJOBS} ]]; then
        wait -n
        active=$((active - 1))
    fi
    sleep 3
done
wait

echo "[$(date '+%T')] all runs finished — consolidating checkpoints for ${EXP_GROUP}/${SIDE} ..."
bash "${REPO}/scripts/overcooked/june_new_exp/consolidate_checkpoints.sh" "${EXP_GROUP}" "${SIDE}"

echo "=========================================================================="
echo "RIGHT-side FC baselines (${EXP_GROUP}) — all runs complete.  $(date '+%F %T')"
echo "=========================================================================="
