#!/bin/bash
# Train a SIDE-AGNOSTIC PACE L2 on FC: one policy that plays BOTH seats, against a
# pool of BOTH-SIDES L1 partners (L1_FC_bothsides_p025). Side-agnostic, so it lives at
# train/ root like train_L1_bothsides.sh.
#
# --both-sides: ego seat fixed per env process (even procs LEFT, odd RIGHT); the
# partner takes the opposite seat via Overcooked_SingleEnv.set_id. Verified 2026-10-20
# on an instrumented smoke with neural partners: each partner object stays on one seat.
#
# Hyperparameters = the PACE FC L2 (L2_left_big, Feb 2026): lr 3e-4, 1000 x 15 procs,
# 60M steps, history 5 / meta-episode, no peer-id head. Partners loaded by the
# --use-l1-partners custom-seeds loader from L1_BASE_DIR/seed{N}/ppo/L1_CKPT.
#
# Pilot launched 2026-10-20 12:xx (user decisions): train seeds 1,4,5 / eval 2,3 from a
# FROZEN snapshot of the still-training both-sides L1s
# (june_new_exp/L1_FC_bothsides_p025_snap_20261020, steps in its README.txt), with
# --feed-partner-history ON (partners get their history as in their own training).
#
# Usage:
#   SEEDS="1" GPUS="4" bash scripts/overcooked/june_new_exp/train/train_L2_bothsides_FC.sh
# Env: SEEDS, GPUS, EXP_TAG, L1_BASE_DIR, L1_CKPT, TRAIN_SEEDS, EVAL_SEEDS,
#      NUM_ENV_STEPS (60000000), MAX_ATTEMPTS (3), EXTRA_ARGS, FEED_HISTORY (1).
# Per-seed retry up to MAX_ATTEMPTS; completion judged by the NUM_ENV_STEPS ckpt, not rc;
# consolidates ONCE at the end. Launch seeds staggered (see train_L2_FI.sh) if >1.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO}"
# Offline like the other june_new_exp launchers. Online mode made wandb.init call the W&B
# server (upsert_run -> server_settings_introspection), which returned nothing on this box
# and killed the run with "UsageError: 'NoneType' object has no attribute 'get'" (2026-10-20).
export WANDB_MODE=${WANDB_MODE:-offline}
# train_.py sets torch.set_num_threads(1) in the MAIN process only; forkserver env workers do not
# inherit it. With --feed-partner-history each worker spun up a 57-thread torch pool (measured
# 2026-10-20: ~190% CPU per worker, 19 workers) on a box with 56 physical cores. OMP_NUM_THREADS
# is inherited by the workers and caps that at 1 thread.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SEEDS=${SEEDS:-"1"}
GPUS=${GPUS:-"4"}
EXP_TAG=${EXP_TAG:-L2_FC_bothsides_p025}
L1_BASE_DIR=${L1_BASE_DIR:-${REPO}/logs/Overcooked/june_new_exp/L1_FC_bothsides_p025_snap_20261020}
L1_CKPT=${L1_CKPT:-snapshot_20261020.pt}
TRAIN_SEEDS=${TRAIN_SEEDS:-"1,4,5"}
EVAL_SEEDS=${EVAL_SEEDS:-"2,3"}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-60000000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
EXTRA_ARGS=${EXTRA_ARGS:-}
FEED_HISTORY=${FEED_HISTORY:-1}
ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml}
[[ "${FEED_HISTORY}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --feed-partner-history"

TRAIN_POOL_SIZE=$(echo "${TRAIN_SEEDS}" | tr ',' '\n' | grep -c .)
EVAL_POOL_SIZE=$(echo "${EVAL_SEEDS}" | tr ',' '\n' | grep -c .)
LOG_ROOT="logs/Overcooked/june_new_exp/${EXP_TAG}"
mkdir -p "${LOG_ROOT}"
read -r -a GPU_ARR <<< "${GPUS}"

echo "=========================================================================="
echo "L2 FC BOTH-SIDES (PACE)  steps=${NUM_ENV_STEPS}"
echo "  Tag:         ${EXP_TAG}"
echo "  L1 pool:     ${L1_BASE_DIR}  (ckpt ${L1_CKPT})"
echo "  Train seeds: ${TRAIN_SEEDS}  (pool size ${TRAIN_POOL_SIZE})"
echo "  Eval seeds:  ${EVAL_SEEDS}   (pool size ${EVAL_POOL_SIZE})"
echo "  Feed partner history: ${FEED_HISTORY}   extra: ${EXTRA_ARGS}"
echo "  Seeds: ${SEEDS}    GPUs: ${GPUS}"
echo "  Started: $(date '+%F %T')"
echo "=========================================================================="

done_ckpt() { [[ -f "logs/Overcooked/ppo_june_new_exp/${EXP_TAG}/seed$1_seed$1/ppo/${NUM_ENV_STEPS}.pt" ]] \
           || [[ -f "logs/Overcooked/june_new_exp/${EXP_TAG}/seed$1/ppo/${NUM_ENV_STEPS}.pt" ]]; }

run_seed() {
    local seed=$1 gpu_id=$2 attempt=1 numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi
    local log_file="${LOG_ROOT}/training_seed${seed}.log"
    while [[ ${attempt} -le ${MAX_ATTEMPTS} ]]; do
        if done_ckpt "${seed}"; then echo "[$(date '+%T')] seed${seed}: already complete"; return 0; fi
        echo "[$(date '+%T')] START seed${seed} (GPU=${gpu_id} NUMA=${numa_node}) attempt ${attempt}/${MAX_ATTEMPTS}"
        # keep every attempt's output (attempt 1's error is otherwise lost on retry)
        [[ -f "${log_file}" ]] && mv "${log_file}" "${log_file}.attempt$((attempt-1))"
        # a previous attempt that died before saving anything leaves a raw dir behind, and
        # train_.py then blocks on an interactive "overwrite?" prompt. Clear it ONLY if it
        # holds no checkpoint at all.
        local raw_dir="logs/Overcooked/ppo_june_new_exp/${EXP_TAG}/seed${seed}_seed${seed}"
        if [[ -d "${raw_dir}" ]] && ! ls "${raw_dir}"/ppo/*.pt >/dev/null 2>&1; then
            echo "[$(date '+%T')] clearing checkpoint-less stale dir ${raw_dir}"
            rm -rf "${raw_dir}"
        fi
        CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
        python train_.py --env-name Overcooked \
          --env_config "${ENV_CONFIG}" \
          --both-sides --player-id 0 --recipe-type cross \
          --algo ppo --num-steps 1000 --num-processes 15 --use-gae --eps 1e-8 --use-proper-time-limit \
          --lr 3e-4 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
          --max-grad-norm 15.0 \
          --num-env-steps ${NUM_ENV_STEPS} \
          --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
          --exp-name "june_new_exp/${EXP_TAG}/seed${seed}" \
          --latent-training --e2e-obj \
          --deterministic-latent \
          --use-latent-critic \
          --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
          --latent-dim 128 --kl-coef 0 \
          --hidden-dims 128 128 --act-func relu \
          --train-pool-size ${TRAIN_POOL_SIZE} --eval-pool-size ${EVAL_POOL_SIZE} \
          --rule-based-opponents 0 \
          --use-l1-partners \
          --l1-partner-base-dir "${L1_BASE_DIR}" \
          --l1-partner-ckpt "${L1_CKPT}" \
          --l1-train-seeds "${TRAIN_SEEDS}" \
          --l1-eval-seeds "${EVAL_SEEDS}" \
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
          ${EXTRA_ARGS} \
          ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log_file}" 2>&1
        local rc=$?
        if done_ckpt "${seed}"; then
            echo "[$(date '+%T')] DONE seed${seed} (${NUM_ENV_STEPS} ckpt present, rc=${rc})"; return 0
        fi
        echo "[$(date '+%T')] seed${seed} exited rc=${rc} with no ${NUM_ENV_STEPS} ckpt — see ${log_file}"
        attempt=$((attempt + 1))
    done
    echo "[$(date '+%T')] GAVE UP on seed${seed} after ${MAX_ATTEMPTS} attempts"; return 1
}

i=0
for seed in ${SEEDS}; do
    gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
    run_seed "${seed}" "${gpu}" &
    i=$((i + 1))
    sleep 5
done
wait

echo "[$(date '+%T')] all seeds terminal; consolidating once"
bash "${REPO}/scripts/overcooked/june_new_exp/consolidate_checkpoints.sh" "${EXP_TAG}" "" 2>/dev/null \
  || echo "[consolidate] skipped/failed — ckpts remain under logs/Overcooked/ppo_june_new_exp/${EXP_TAG}/seed{N}_seed{N}"
echo "L2 FC both-sides complete. $(date '+%F %T')"
