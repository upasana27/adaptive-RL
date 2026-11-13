#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Train L1 LEFT FI agents at p=0.5 L0 stochasticity — 6-recipe disjoint domain.
#
# Same domain / hyperparameters as train_L1_FI.sh (which is frozen at p=0.0);
# this variant differs only in:
#   * --p ${P}         (default 0.5 — stochastic L0 partners)
#   * output tag       (L1_FI_p05, so the p=0.0 checkpoints are NOT clobbered)
#   * CONCURRENCY CAP  (MAXJOBS) — the original launched all 30 runs at once,
#                      which (together with the right side) drove the machine
#                      into swap and mass-killed every run via ConnectionReset.
#                      Here at most MAXJOBS run concurrently.
#
# Output:
#   stdout logs -> logs/Overcooked/june_new_exp/L1_FI_p05/left/{TYPE}_seed{N}.log
#   checkpoints -> logs/Overcooked/ppo_june_new_exp/L1_FI_p05/left/{TYPE}/seed{N}_seed{N}/ppo/
#                  (consolidated post-run to june_new_exp/L1_FI_p05/left/{TYPE}/seed{N}/ppo/)
#
# Usage (launch durably so batching survives disconnect):
#   setsid nohup bash scripts/overcooked/june_new_exp/train/left/train_L1_FI_p05.sh \
#       > logs/Overcooked/june_new_exp/L1_FI_p05/left/_launcher.log 2>&1 < /dev/null &
#
# Env overrides: P (0.5), SEEDS, TYPES, GPUS ("5 6 7"), MAXJOBS (6),
#                EXP_TAG (L1_FI_p05), SIDE (left) + PLAYER_ID (0).
#
# SIDE/PLAYER_ID override lets this same batched launcher drive the RIGHT side
# (same precedent as the SIDE=left override on train_baselines_FC_right.sh).
# Defaults are unchanged, so a bare invocation reproduces the original left/p=0.5 run.
#   RIGHT side, p=0.25, PACE arch, 8 seeds/type:
#     SIDE=right PLAYER_ID=1 P=0.25 EXP_TAG=L1_FI_pace_p025 \
#     SEEDS="1 2 3 4 5 6 7 8" GPUS="1 2 3 4 5 6" MAXJOBS=6 \
#     setsid nohup bash scripts/overcooked/june_new_exp/train/left/train_L1_FI_p05.sh &
# NOTE: keep the SUM of MAXJOBS across concurrent launchers <= ~8. Each run forks
# 13 procs; the 2026-07-10 FI pool died when 60 simultaneous runs (~780 procs)
# exhausted RAM and the SubprocVecEnv workers were killed (80x ConnectionResetError).

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

# wandb online mode deadlocks the training loop when the backend returns 403
# (its console-capture blocks the next print()). Run fully offline; local
# wandb dirs are still written and can be synced later.
export WANDB_MODE=offline

set -uo pipefail

SIDE=${SIDE:-left}
PLAYER_ID=${PLAYER_ID:-0}
P=${P:-0.5}
EXP_TAG=${EXP_TAG:-L1_FI_p05}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
TYPES=${TYPES:-"R1R2 R3R4 R5R6"}
GPUS=${GPUS:-"5 6 7"}
MAXJOBS=${MAXJOBS:-6}
# ENV_CONFIG override: lets the same launcher train on a horizon variant
# (fc_small_6recipe_h60/_h80.yaml). Default = the canonical horizon-40 config.
# NOTE: --horizon-override is a DEAD FLAG in this repo (parsed, never read) —
# horizon MUST be changed via the env config.
ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_6recipe.yaml}

read -ra SEED_ARR <<< "${SEEDS}"
read -ra TYPE_ARR <<< "${TYPES}"
read -ra GPU_ARR  <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}

REPO=${PACE_ROOT}
cd "${REPO}"
LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_TAG}/${SIDE}
mkdir -p "${LOG_ROOT}"

echo "=========================================================================="
echo "L1 ${SIDE^^} FI @ p=${P}  (6-recipe disjoint, batched)"
echo "  Tag:      ${EXP_TAG}"
echo "  Types:    ${TYPES}"
echo "  Seeds:    ${SEEDS}"
echo "  GPUs:     ${GPUS}   MAXJOBS=${MAXJOBS}"
echo "  Env cfg:  ${ENV_CONFIG}"
echo "  Total:    $(( ${#TYPE_ARR[@]} * ${#SEED_ARR[@]} )) runs (30M env steps each)"
echo "  Started:  $(date '+%F %T')"
echo "=========================================================================="

run_one() {
    local seed=$1 agent_type=$2 gpu_id=$3
    local exp_dir="june_new_exp/${EXP_TAG}/${SIDE}/${agent_type}/seed${seed}"
    local log_file="${LOG_ROOT}/${agent_type}_seed${seed}.log"
    local numa_node
    if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi

    echo "[$(date '+%T')] START ${agent_type}_seed${seed} (GPU=${gpu_id} NUMA=${numa_node} p=${P})"
    CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config "${ENV_CONFIG}" \
      --recipe-domain 6recipe --l0-pair-type ${agent_type} \
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
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log_file}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        # CONSOLIDATE=0 defers consolidation to the caller. Per-job consolidation
        # is RACY: consolidate_checkpoints.sh mv's EVERY seed*_seed* dir under the
        # tag/side, including dirs of still-running jobs, which yanks their wandb
        # dir and kills them at exit with FileNotFoundError (rc=255). That is what
        # happened to 21 of 27 runs in the 2026-10-02 p=0.25 pool.
        if [[ "${CONSOLIDATE:-1}" == "1" ]]; then
            bash "${REPO}/scripts/overcooked/june_new_exp/consolidate_checkpoints.sh" "${EXP_TAG}" "${SIDE}" >> "${log_file}" 2>&1
        fi
        echo "[$(date '+%T')] DONE  ${agent_type}_seed${seed} (rc=0)"
    else
        echo "[$(date '+%T')] FAIL  ${agent_type}_seed${seed} (rc=${rc}) — see ${log_file}"
    fi
}

# Build job list (type-major, same order as the original script).
JOBS=()
for agent_type in "${TYPE_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
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
echo "All L1 ${SIDE^^} FI @ p=${P} runs complete.  $(date '+%F %T')"
echo "=========================================================================="
