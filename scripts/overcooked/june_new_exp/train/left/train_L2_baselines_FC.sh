#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Train each baseline as an L2 agent against ITS OWN L1 pool (LEFT side, player-id 0).
#
# Design (the training regime is the contribution, applied per-architecture end-to-end):
#   For each method M, L2-M (M's architecture) is trained against the L1-M pool
#   (M's architecture, trained vs L0 by train_baselines_FC_right.sh). Everything for a
#   method is w.r.t. that method itself -- L2-LiAM trains against L1-LiAM, never against
#   PACE's L1 pool. Because L2-M and L1-M share an architecture, the partner is loaded
#   with exactly the flags it was trained with (no cross-arch mismatch).
#
# PILOT pool: 4 L1 seeds per method -> 3 train (1,2,3) + 1 eval (4).
# PACE is intentionally left out for now (its L1 pool is handled separately).
#
# Per-method L1 pool base dir (passed via --l1-partner-base-dir):
#   ${L1_PARTNER_ROOT}/${method}  with seed{N}/ppo/30000000.pt
#   (loader also accepts the un-consolidated seed{N}_seed{N}/ layout).
# Baselines run EXACTLY as their published architectures define them -- per-method flags
# copied VERBATIM from train/right/train_baselines_FC_right.sh (frozen {liam,liamx,lili,
# generalist}.sh). Native baseline optimizer kept (lr 1e-3, 12 procs), 60M L2 steps.
#
# Output (stdout):     logs/Overcooked/june_new_exp/L2_baselines_FC/left/{METHOD}_seed{N}.log
# Output (checkpoints): logs/Overcooked/june_new_exp/L2_baselines_FC/left/{METHOD}/seed{N}/ppo/
#
# Smoke test (validate the per-method loader end-to-end, uses existing L1 seed1 only):
#   NUM_ENV_STEPS=60000 SAVE_INTERVAL=48000 EVAL_INTERVAL=24000 \
#   EXP_GROUP=L2_baselines_FC_smoke METHODS=LiAM SEEDS=1 MAXJOBS=1 GPUS="4" \
#   L1_PARTNER_ROOT=logs/Overcooked/ppo_june_new_exp/baselines_FC/right \
#   L1_TRAIN_SEEDS=1 L1_EVAL_SEEDS=1 TRAIN_POOL_SIZE=1 EVAL_POOL_SIZE=1 \
#   bash scripts/overcooked/june_new_exp/train/left/train_L2_baselines_FC.sh
#
# Full pilot (after L1 seeds 1-4 exist + consolidated), all 4 baselines x seeds:
#   setsid nohup bash scripts/overcooked/june_new_exp/train/left/train_L2_baselines_FC.sh \
#       > logs/Overcooked/june_new_exp/L2_baselines_FC/left/_launcher.log 2>&1 < /dev/null &

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline
set -uo pipefail

REPO=${PACE_ROOT}
cd "${REPO}"

SIDE=left
PLAYER_ID=0
P=${P:-0.25}
LR=${LR:-1e-3}
NUM_PROCS=${NUM_PROCS:-12}
SEEDS=${SEEDS:-"1 2 3"}
METHODS=${METHODS:-"LiAM LiAMx LiLi Generalist"}   # PACE left out for now
GPUS=${GPUS:-"4 0 5 6 7"}
MAXJOBS=${MAXJOBS:-4}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-60000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
EXP_GROUP=${EXP_GROUP:-L2_baselines_FC}

# Per-method L1 pool (each method trains against its OWN L1-M pool).
L1_PARTNER_ROOT=${L1_PARTNER_ROOT:-logs/Overcooked/june_new_exp/baselines_FC/right}
# BOTH_SIDES=1 (2026-10-26): side-agnostic L2 via --both-sides (ego seat alternates per env process),
# against a both-sides L1 pool (set L1_PARTNER_ROOT to .../baselines_FC_bothsides_p025/bothsides).
# Forces SIDE=bothsides, PLAYER_ID=0. EXTRA_ARGS is appended verbatim (e.g. --feed-partner-history).
BOTH_SIDES=${BOTH_SIDES:-0}
BOTH_SIDES_FLAG=""
if [[ "${BOTH_SIDES}" == "1" ]]; then SIDE=bothsides; PLAYER_ID=0; BOTH_SIDES_FLAG="--both-sides"; fi
EXTRA_ARGS=${EXTRA_ARGS:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # caps torch threads in the forkserver env workers
L1_TRAIN_SEEDS=${L1_TRAIN_SEEDS:-"1,2,3"}
L1_EVAL_SEEDS=${L1_EVAL_SEEDS:-"4"}
TRAIN_POOL_SIZE=${TRAIN_POOL_SIZE:-3}
EVAL_POOL_SIZE=${EVAL_POOL_SIZE:-1}

LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_GROUP}/${SIDE}
mkdir -p "${LOG_ROOT}"

read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}

echo "=========================================================================="
echo "LEFT-side L2 baselines vs OWN L1 pool  (group=${EXP_GROUP})"
echo "  Methods: ${METHODS}    Seeds: ${SEEDS}    GPUs: ${GPUS}   MAXJOBS=${MAXJOBS}"
echo "  L1 pool: ${L1_PARTNER_ROOT}/{METHOD} (train ${L1_TRAIN_SEEDS} / eval ${L1_EVAL_SEEDS})"
echo "  Regime:  fc_small_test, player-id ${PLAYER_ID}, lr ${LR}, ${NUM_PROCS} procs, --p ${P}, ${NUM_ENV_STEPS} steps"
echo "  Output:  ${LOG_ROOT}"
echo "  Started: $(date '+%F %T')"
echo "=========================================================================="

# Per-method architecture flags, copied VERBATIM from
# scripts/overcooked/june_new_exp/train/right/train_baselines_FC_right.sh::method_flags.
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
    local l1_base="${L1_PARTNER_ROOT}/${method}"
    local numa_node
    if [[ "${gpu}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "[$(date '+%T')] START ${method}_seed${seed} (GPU=${gpu} NUMA=${numa_node})  L1<-${l1_base}"
    CUDA_VISIBLE_DEVICES=${gpu} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config ./environment/overcooked/config/fc_small_test.yaml \
      --player-id ${PLAYER_ID} ${BOTH_SIDES_FLAG} --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes ${NUM_PROCS} --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr ${LR} --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps ${NUM_ENV_STEPS} \
      --save-interval ${SAVE_INTERVAL} --log-interval 10000 --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
      --exp-name "${exp_dir}" \
      --train-pool-size ${TRAIN_POOL_SIZE} --eval-pool-size ${EVAL_POOL_SIZE} \
      --rule-based-opponents 0 \
      --use-l1-partners \
      --l1-partner-base-dir "${l1_base}" \
      --l1-train-seeds ${L1_TRAIN_SEEDS} \
      --l1-eval-seeds ${L1_EVAL_SEEDS} \
      --seed ${seed} \
      --p ${P} \
      $(method_flags "${method}") \
      ${EXTRA_ARGS} \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log_file}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "[$(date '+%T')] DONE  ${method}_seed${seed} (rc=0)"
    else
        echo "[$(date '+%T')] FAIL  ${method}_seed${seed} (rc=${rc}) — see ${log_file}"
    fi
}

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
echo "LEFT-side L2 baselines (${EXP_GROUP}) — all runs complete.  $(date '+%F %T')"
echo "=========================================================================="
