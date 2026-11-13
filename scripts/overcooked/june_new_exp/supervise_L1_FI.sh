#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Resilient supervisor for an FI L1 pool: keeps the pool running until every
# job has a 30M checkpoint, relaunching any job whose process has died.
#
# Why this exists: the 2026-10-02 pool was launched fire-and-forget; a race in
# per-job consolidation killed 21/27 runs at exit and nobody relaunched anything
# for four days. This script owns the whole queue instead.
#
# Mechanism (no duplicated training command): each job is started by invoking the
# existing batched launcher train/left/train_L1_FI_p05.sh with SEEDS/TYPES narrowed
# to that ONE job, MAXJOBS=1 and CONSOLIDATE=0. Completion = 30000000.pt present
# (raw ppo_june_new_exp/ or consolidated june_new_exp/ location), NOT the exit code,
# so a cosmetic crash after the final save does not trigger a rerun. Consolidation
# happens exactly once, after every job is complete.
#
# Usage (durable):
#   P=0.0 EXP_TAG=L1_FI_pace_p0 \
#   setsid nohup bash scripts/overcooked/june_new_exp/supervise_L1_FI.sh \
#     > logs/Overcooked/june_new_exp/L1_FI_pace_p0/_supervisor.log 2>&1 < /dev/null &
#
# Env: P (0.0), EXP_TAG (L1_FI_pace_p0), TYPES ("R1R2 R3R4 R5R6"),
#      SEEDS_RIGHT ("1 2 3 4 5 6 7 8"), SEEDS_LEFT ("1"), GPUS ("0 1 2 3 4 5 6 7"),
#      MAXJOBS (8), MAX_ATTEMPTS (3), POLL (120 s).
# Safe to kill and restart at any time: it re-derives state from the filesystem.

set -uo pipefail
REPO=${PACE_ROOT}
cd "${REPO}"

P=${P:-0.0}
EXP_TAG=${EXP_TAG:-L1_FI_pace_p0}
TYPES=${TYPES:-"R1R2 R3R4 R5R6"}
SEEDS_RIGHT=${SEEDS_RIGHT:-"1 2 3 4 5 6 7 8"}
SEEDS_LEFT=${SEEDS_LEFT-"1"}   # ${X-d}: empty string means NO left runs
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
MAXJOBS=${MAXJOBS:-8}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
POLL=${POLL:-120}
LAUNCHER=scripts/overcooked/june_new_exp/train/left/train_L1_FI_p05.sh

read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_TAG}
STATE=${LOG_ROOT}/_supervisor
mkdir -p "${LOG_ROOT}/left" "${LOG_ROOT}/right" "${STATE}"
export PYTHONUNBUFFERED=1   # timely logs; no effect on training

# Job list: LEFT control first (finishes early, early signal), then RIGHT type-major.
JOBS=()
for t in ${TYPES}; do for s in ${SEEDS_LEFT};  do JOBS+=("left ${t} ${s}");  done; done
for t in ${TYPES}; do for s in ${SEEDS_RIGHT}; do JOBS+=("right ${t} ${s}"); done; done

ts() { date '+%F %T'; }
is_complete() {  # side type seed
    [[ -f "logs/Overcooked/ppo_june_new_exp/${EXP_TAG}/$1/$2/seed$3_seed$3/ppo/30000000.pt" ]] || \
    [[ -f "logs/Overcooked/june_new_exp/${EXP_TAG}/$1/$2/seed$3/ppo/30000000.pt" ]]
}
is_running() {  # side type seed — exact match on the exp-name arg
    pgrep -f -- "exp-name june_new_exp/${EXP_TAG}/$1/$2/seed$3 --latent" > /dev/null
}
attempts() { cat "${STATE}/$1_$2_seed$3.attempts" 2>/dev/null || echo 0; }
bump()     { echo $(( $(attempts "$@") + 1 )) > "${STATE}/$1_$2_seed$3.attempts"; }
launch() {  # side type seed gpu
    local side=$1 typ=$2 seed=$3 gpu=$4 pid
    if [[ "${side}" == left ]]; then pid=0; else pid=1; fi
    bump "${side}" "${typ}" "${seed}"
    echo "[$(ts)] LAUNCH ${side}/${typ}/seed${seed} on GPU ${gpu} (attempt $(attempts "${side}" "${typ}" "${seed}")/${MAX_ATTEMPTS})"
    SIDE=${side} PLAYER_ID=${pid} P=${P} EXP_TAG=${EXP_TAG} SEEDS="${seed}" TYPES="${typ}" \
    GPUS="${gpu}" MAXJOBS=1 CONSOLIDATE=0 \
    setsid nohup bash "${LAUNCHER}" >> "${LOG_ROOT}/${side}/_launcher.log" 2>&1 < /dev/null &
}

echo "=================================================================="
echo "SUPERVISOR ${EXP_TAG}  p=${P}  jobs=${#JOBS[@]}  MAXJOBS=${MAXJOBS}  GPUs=${GPUS}"
echo "  started $(ts)"
echo "=================================================================="

gpu_idx=0
while true; do
    n_done=0; n_run=0; n_dead=0; pending=()
    for job in "${JOBS[@]}"; do
        read -r side typ seed <<< "${job}"
        if is_complete "${side}" "${typ}" "${seed}"; then n_done=$((n_done+1))
        elif is_running "${side}" "${typ}" "${seed}"; then n_run=$((n_run+1))
        elif [[ $(attempts "${side}" "${typ}" "${seed}") -ge ${MAX_ATTEMPTS} ]]; then n_dead=$((n_dead+1))
        else pending+=("${job}")
        fi
    done

    # (Re)launch pending jobs up to the concurrency cap.
    for job in "${pending[@]}"; do
        [[ ${n_run} -ge ${MAXJOBS} ]] && break
        read -r side typ seed <<< "${job}"
        a=$(attempts "${side}" "${typ}" "${seed}")
        [[ ${a} -gt 0 ]] && echo "[$(ts)] RELAUNCH ${side}/${typ}/seed${seed}: process gone, no 30M ckpt (prior attempts=${a}) — see ${LOG_ROOT}/${side}/${typ}_seed${seed}.log"
        launch "${side}" "${typ}" "${seed}" "${GPU_ARR[$((gpu_idx % NUM_GPUS))]}"
        gpu_idx=$((gpu_idx+1)); n_run=$((n_run+1)); sleep 5
    done

    echo "[$(ts)] status: done=${n_done} running=${n_run} pending=$(( ${#JOBS[@]} - n_done - n_run - n_dead )) gave_up=${n_dead} / ${#JOBS[@]}"
    if [[ $(( n_done + n_dead )) -eq ${#JOBS[@]} ]]; then break; fi
    sleep "${POLL}"
done

echo "[$(ts)] all jobs terminal (done=${n_done}, gave_up=${n_dead}); consolidating ONCE per side"
for side in left right; do
    bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh "${EXP_TAG}" "${side}"
done
if [[ ${n_dead} -gt 0 ]]; then
    echo "[$(ts)] WARNING: ${n_dead} job(s) exhausted ${MAX_ATTEMPTS} attempts — inspect ${STATE}/*.attempts and the run logs"
fi
echo "[$(ts)] SUPERVISOR EXIT"
