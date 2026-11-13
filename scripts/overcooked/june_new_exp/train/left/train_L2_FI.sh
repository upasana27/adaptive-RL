#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# L2 FI (Exp 2) — L2 LEFT vs the disjoint-capability L1 RIGHT pool, 6-recipe domain.
#
# Modeled on train_L2_PC.sh (same PACE L2 hyperparameters: lr 3e-4, 15 procs, 60M)
# but swaps the SPECIALIZED loader for the DISJOINT one and the 3-recipe env for
# fc_small_6recipe.yaml.
#
# Partner pool (2026-10 PACE FI L1 pool, p=0.0, train 0.84 / eval 0.83):
#   june_new_exp/L1_FI_pace_p0/right/{R1R2,R3R4,R5R6}/seed{N}/ppo/30000000.pt
# Split is 5 train + 2 held-out per type (the 2 recipe-COLLAPSED seeds —
# R1R2/seed5 and R3R4/seed1 — are EXCLUDED from the pool entirely per user
# decision 2026-10-16; R5R6/seed8 is a spare):
#   Train (5/type): R1R2 1,2,3,4,6 | R3R4 2,3,4,5,6 | R5R6 1,2,3,4,5  -> 15 partners
#   Held out (2/type): R1R2 7,8    | R3R4 7,8       | R5R6 6,7        ->  6 partners
# NOTE: EVAL_SPEC here is the DURING-TRAINING monitor. The definitive held-out
# evaluation is a separate pass with the 10x5 protocol (--test-pass 10
# --test-episodes 5), per the eval-regime rule in CLAUDE.md.
#
# Resilient: each seed is retried up to MAX_ATTEMPTS if it exits without a 60M
# checkpoint. Completion is judged by the checkpoint, NOT the exit code, because a
# finished run can still exit non-zero on wandb teardown. Consolidates ONCE at the end
# (per-job consolidation races and kills live runs — see the 2026-10-02 incident).
#
# Usage (durable):
#   setsid nohup bash scripts/overcooked/june_new_exp/train/left/train_L2_FI.sh \
#     > logs/Overcooked/june_new_exp/L2_FI/left/_launcher.log 2>&1 < /dev/null &
#   SMOKE=1 SEEDS=99 GPUS=0 bash .../train_L2_FI.sh     # 150k-step pool-load validation
#
# Env: SEEDS ("1 2 3"), GPUS ("0 1 2"), EXP_TAG (L2_FI), L1_BASE_DIR, TRAIN_SPEC,
#      EVAL_SPEC, TRAIN_POOL_SIZE (15), EVAL_POOL_SIZE (6), NUM_ENV_STEPS (60000000),
#      MAX_ATTEMPTS (3), EXTRA_ARGS, SMOKE (0).

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=${WANDB_MODE:-offline}   # online mode deadlocks the loop on a 403
# Cap torch threads in the forkserver env workers (train_.py's set_num_threads(1) covers the main
# process only). Measured 2026-10-20 on the --feed-partner-history run: 57 threads / ~190% CPU per
# worker vs 2 threads / 50% without the feed. Affects future launches only.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
set -uo pipefail

REPO=${PACE_ROOT}
cd "${REPO}"

SEEDS=${SEEDS:-"1 2 3"}
GPUS=${GPUS:-"0 1 2"}
EXP_TAG=${EXP_TAG:-L2_FI}
SIDE=left
PLAYER_ID=0
L1_BASE_DIR=${L1_BASE_DIR:-${REPO}/logs/Overcooked/june_new_exp/L1_FI_pace_p0/right}
TRAIN_SPEC=${TRAIN_SPEC:-'R1R2:1,2,3,4,6;R3R4:2,3,4,5,6;R5R6:1,2,3,4,5'}
EVAL_SPEC=${EVAL_SPEC:-'R1R2:7,8;R3R4:7,8;R5R6:6,7'}
TRAIN_POOL_SIZE=${TRAIN_POOL_SIZE:-15}
EVAL_POOL_SIZE=${EVAL_POOL_SIZE:-6}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-60000000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
EXTRA_ARGS=${EXTRA_ARGS:-}
ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_6recipe.yaml}
# IMPORTANT: match the horizon the PARTNERS were trained at. The p=0.25 pool was
# trained at H=80 (fc_small_6recipe_h80.yaml); running it inside H=40 episodes is a
# distribution shift for the partners. --horizon-override is a DEAD FLAG in this repo.
SMOKE=${SMOKE:-0}
if [[ "${SMOKE}" == "1" ]]; then NUM_ENV_STEPS=150000; EXP_TAG="${EXP_TAG}_smoke"; MAX_ATTEMPTS=1; fi

read -ra SEED_ARR <<< "${SEEDS}"
read -ra GPU_ARR  <<< "${GPUS}"
LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_TAG}/${SIDE}
mkdir -p "${LOG_ROOT}"

echo "=========================================================================="
echo "L2 FI ${SIDE^^} (6-recipe disjoint)  player-id=${PLAYER_ID}  steps=${NUM_ENV_STEPS}"
echo "  Tag:        ${EXP_TAG}"
echo "  L1 pool:    ${L1_BASE_DIR}"
echo "  Train spec: ${TRAIN_SPEC}   (pool size ${TRAIN_POOL_SIZE})"
echo "  Eval spec:  ${EVAL_SPEC}    (pool size ${EVAL_POOL_SIZE})"
echo "  Seeds:      ${SEEDS}    GPUs: ${GPUS}"
echo "  Env cfg:    ${ENV_CONFIG}"
echo "  Started:    $(date '+%F %T')"
echo "=========================================================================="

done_ckpt() { [[ -f "logs/Overcooked/ppo_june_new_exp/${EXP_TAG}/${SIDE}/seed$1_seed$1/ppo/${NUM_ENV_STEPS}.pt" ]] \
           || [[ -f "logs/Overcooked/june_new_exp/${EXP_TAG}/${SIDE}/seed$1/ppo/${NUM_ENV_STEPS}.pt" ]]; }

run_seed() {
    local seed=$1 gpu_id=$2 attempt=1 numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi
    local log_file="${LOG_ROOT}/training_seed${seed}.log"
    while [[ ${attempt} -le ${MAX_ATTEMPTS} ]]; do
        if done_ckpt "${seed}"; then echo "[$(date '+%T')] seed${seed}: already complete"; return 0; fi
        echo "[$(date '+%T')] START seed${seed} (GPU=${gpu_id} NUMA=${numa_node}) attempt ${attempt}/${MAX_ATTEMPTS}"
        CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
        python train_.py --env-name Overcooked \
          --env_config "${ENV_CONFIG}" \
          --recipe-domain 6recipe \
          --player-id ${PLAYER_ID} --recipe-type cross \
          --algo ppo --num-steps 1000 --num-processes 15 --use-gae --eps 1e-8 --use-proper-time-limit \
          --lr 3e-4 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
          --max-grad-norm 15.0 \
          --num-env-steps ${NUM_ENV_STEPS} \
          --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
          --exp-name "june_new_exp/${EXP_TAG}/${SIDE}/seed${seed}" \
          --latent-training --e2e-obj \
          --deterministic-latent \
          --use-latent-critic \
          --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
          --latent-dim 128 --kl-coef 0 \
          --hidden-dims 128 128 --act-func relu \
          --train-pool-size ${TRAIN_POOL_SIZE} --eval-pool-size ${EVAL_POOL_SIZE} \
          --rule-based-opponents 0 \
          --use-l1-disjoint \
          --l1-disjoint-base-dir "${L1_BASE_DIR}" \
          --l1-disjoint-train "${TRAIN_SPEC}" \
          --l1-disjoint-eval  "${EVAL_SPEC}" \
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

# STARTUP STAGGER (default on). Launching L2 runs within seconds of each other
# deadlocks all but the first: 2026-10-16 seeds 2+3 wedged with frozen logs at
# identical offsets and CPU pinned ~6% while seed1 (first out) trained fine.
# The race is in CUDA/forkserver init, so we serialize STARTUP only — each seed
# waits until its predecessor prints a real update, then launches. Training is
# still fully concurrent. STAGGER=0 restores the old simultaneous launch.
STAGGER=${STAGGER:-1}
STAGGER_TIMEOUT=${STAGGER_TIMEOUT:-1800}

_wait_until_training() {
    local seed=$1 log="${LOG_ROOT}/training_seed$1.log" t=0
    while [[ ${t} -lt ${STAGGER_TIMEOUT} ]]; do
        grep -qE "num timesteps" "${log}" 2>/dev/null && { echo "[$(date '+%T')] seed${seed} is training; releasing next launch"; return 0; }
        ps -eo cmd | grep -q "[t]rain_\.py.*${EXP_TAG}/${SIDE}/seed${seed} " || { echo "[$(date '+%T')] seed${seed} died during startup; releasing next launch"; return 1; }
        sleep 20; t=$((t + 20))
    done
    echo "[$(date '+%T')] WARNING seed${seed} not training after ${STAGGER_TIMEOUT}s (likely the startup deadlock) — releasing next launch anyway"
    return 1
}

i=0
for seed in "${SEED_ARR[@]}"; do
    run_seed "${seed}" "${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}" &
    i=$((i + 1))
    if [[ "${STAGGER}" == "1" && ${i} -lt ${#SEED_ARR[@]} ]]; then
        _wait_until_training "${seed}"
    else
        sleep 5
    fi
done
wait

# CONSOLIDATE=0 when another launcher instance still has seeds running for this
# tag: consolidate_checkpoints.sh mv's EVERY seed*_seed* dir under the tag/side,
# including LIVE ones, which kills them (the 2026-10-02 incident).
if [[ "${CONSOLIDATE:-1}" == "1" ]]; then
    echo "[$(date '+%T')] all seeds terminal; consolidating once"
    bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh "${EXP_TAG}" "${SIDE}"
else
    echo "[$(date '+%T')] all seeds terminal; CONSOLIDATE=0 — consolidate manually when every seed is done"
fi
echo "=========================================================================="
echo "L2 FI ${SIDE^^} complete.  $(date '+%F %T')"
echo "=========================================================================="
