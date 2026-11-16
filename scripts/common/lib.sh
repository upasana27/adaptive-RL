#!/bin/bash
# Shared helpers for every launch script. Sourced by the scripts in scripts/common/, never run directly.
#
# Environment variables honoured everywhere:
#   PACE_ROOT   repository root (default: derived from this file's location)
#   CONDA_SH    conda.sh to source (default: $HOME/anaconda3/etc/profile.d/conda.sh)
#   CONDA_ENV   conda environment to activate (default: pace)
#   WANDB_USER  Weights & Biases entity; unset = no wandb logging
#   SEEDS       space-separated seed list, or N_SEEDS=k for seeds 1..k
#   GPUS        space-separated GPU ids, used round-robin
#   MAXJOBS     maximum number of concurrent runs started by one script
#   SMOKE=1     tiny run for checking that a script launches (24k env steps, checkpoint at 12k)
set -euo pipefail
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PACE_ROOT}"
if [[ -f "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-pace}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

EXP_PREFIX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    NUM_ENV_STEPS=24000; SAVE_INTERVAL=12000; EVAL_INTERVAL=12000; EXP_PREFIX="smoke_"
    TEST_PASS="${TEST_PASS:-1}"; TEST_EPISODES="${TEST_EPISODES:-5}"
fi

# "1 2 3" from SEEDS, or 1..N_SEEDS
expand_seeds() {
    if [[ -n "${N_SEEDS:-}" ]]; then seq -s ' ' 1 "${N_SEEDS}"; else echo "${SEEDS}"; fi
}

# Seat handling. SIDE is left, right or bothsides.
side_flags() {
    case "$1" in
        left)      echo "--player-id 0" ;;
        right)     echo "--player-id 1" ;;
        bothsides) echo "--player-id 0 --both-sides" ;;
        *) echo "SIDE must be left, right or bothsides (got '$1')" >&2; exit 1 ;;
    esac
}
partner_side()   { case "$1" in left) echo right ;; right) echo left ;; bothsides) echo bothsides ;; esac; }
seats_for_side() { case "$1" in left) echo 0 ;; right) echo 1 ;; bothsides) echo "0 1" ;; esac; }

# Architecture flags per method. Identical in every domain; the PPO flags are added by the callers.
#   Ours        latent partner encoder trained end to end through PPO, no auxiliary loss
#   PACE        Ours + peer-identification head and classification reward (the published PACE)
#   LiAM        GRU policy + partner observation/action prediction heads, history reset every episode
#   LiAMx       LiAM with the recurrent state carried across the episodes of a round
#   LiLi        latent encoder + transition-prediction auxiliary loss on the previous episode
#   Generalist  GRU policy over the round, no auxiliary loss
LATENT_COMMON="--latent-training --e2e-obj --deterministic-latent --use-latent-critic \
--pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
--latent-dim 128 --kl-coef 0 --hidden-dims 128 128 --act-func relu \
--history-middle-sampling --merge-encoder-computation \
--history-full-size 5 --history-size 5 --history-use-episodes --include-current-episode --use-meta-episode \
--self-obs-mode --clear-history-on-full --separate-history"
RNN_COMMON="--joint-training --hidden-dims 128 128 --act-func relu \
--recurrent-policy --rnn-hidden-dim 128 --rnn-chunk-length 20 --share-actor-critic --history-size 5"
method_flags() {
    case "$1" in
        Ours)       echo "${LATENT_COMMON}" ;;
        PACE)       echo "${LATENT_COMMON} --auxiliary-policy-cls-coef 1.0 --policy-cls-reward-coef 0.2 \
--policy-cls-reward-type accuracy --policy-cls-reward-mode max_full \
--policy-cls-warmup-steps 1000000 --policy-cls-reward-decay-steps 25000000" ;;
        LiLi)       echo "${LATENT_COMMON} --auxiliary-transition-pred-coef 1.0 --self-action-mode --last-episode-only --collect-next-obs" ;;
        LiAM)       echo "${RNN_COMMON} --all-has-last-action --collect-peer-traj \
--auxiliary-peer-obs-pred-coef 1.0 --auxiliary-peer-act-pred-coef 1.0" ;;
        LiAMx)      echo "${RNN_COMMON} --all-has-last-action --collect-peer-traj \
--auxiliary-peer-obs-pred-coef 1.0 --auxiliary-peer-act-pred-coef 1.0 --use-meta-episode" ;;
        Generalist) echo "${RNN_COMMON} --use-meta-episode" ;;
        *) echo "unknown METHOD '$1' (Ours, PACE, LiAM, LiAMx, LiLi, Generalist)" >&2; exit 1 ;;
    esac
}
# Flags an ego needs at evaluation time (its observation layout).
ego_eval_flags() { case "$1" in LiAM|LiAMx) echo "--all-has-last-action" ;; *) echo "" ;; esac; }
# Learning rate: Ours uses 3e-4 for FC level 1 and every level 2, 1e-3 otherwise; baselines always 1e-3.
default_lr() {  # $1 method, $2 level (L1|L2), $3 domain
    if [[ "$1" == Ours && ( "$2" == L2 || "$3" == FC ) ]]; then echo 3e-4; else echo 1e-3; fi
}

# Checkpoint lookup: resolve_ckpt <run dir without the seed> <seed> <preferred file>
# Tries {dir}/seed{N}/ppo and {dir}/seed{N}_seed{N}/ppo (train_.py's raw layout) with the preferred
# file, then latest.pt and individual_latest.pt. Prints the path; returns 1 if nothing exists.
resolve_ckpt() {
    local base=$1 seed=$2 pref=$3 sub f p
    for sub in "seed${seed}" "seed${seed}_seed${seed}"; do
        for f in "${pref}" latest.pt individual_latest.pt; do
            p="${base}/${sub}/ppo/${f}"
            if [[ -f "${p}" ]]; then echo "${p}"; return 0; fi
        done
    done
    echo "${base}/seed${seed}/ppo/${pref}"; return 1
}
# Count the seeds in a spec such as "R1R2:1,2;R3R4:1,2" or a plain list "1,2,3".
spec_count() { echo "$1" | tr ';' '\n' | sed 's/^[^:]*://' | tr ',' '\n' | grep -c .; }

# GPU round-robin and concurrency.
_GPU_ARR=(${GPUS:-0}); _GPU_I=0
next_gpu() { echo "${_GPU_ARR[$(( _GPU_I % ${#_GPU_ARR[@]} ))]}"; _GPU_I=$((_GPU_I + 1)); }
throttle() { while (( $(jobs -rp | wc -l) >= ${MAXJOBS:-4} )); do sleep 20; done; }
# Runs with neural partner pools must not start at the same moment (startup race in the environment
# workers). Wait until the previous run has printed its first update line, or has exited.
stagger_wait() {  # $1 log file, $2 pid
    local t0=${SECONDS}
    until grep -q "num timesteps" "$1" 2>/dev/null || (( SECONDS - t0 > ${STAGGER_TIMEOUT:-1800} )); do
        kill -0 "$2" 2>/dev/null || return 0
        sleep 15
    done
}
