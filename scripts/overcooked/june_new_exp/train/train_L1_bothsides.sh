#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Train a SIDE-AGNOSTIC L1: one PACE policy that plays BOTH seats.
#
# Uses --both-sides (arguments.py): the ego seat is fixed PER ENV PROCESS
# (even procs = LEFT/player-id 0, odd = RIGHT/player-id 1), giving 50/50 batches
# and a side-consistent meta-history with no mid-episode switching. --player-id is
# ignored for seating, so this script lives at train/ root rather than left/ or right/.
#
# Lives alongside train_L1_FC.sh / train_L1_PC.sh and reuses their exact
# hyperparameters per domain -- the ONLY additions are --both-sides and the tag:
#   FC: lr 3e-4, 3 rule-based L0s, pool-seed 1,            train/eval pool 3
#   PC: lr 1e-3, 2 rule-based L0s, per-type pool-seed,     train/eval pool 2
#       (POOL_SEED: LO_PB=3, LO_TC=1, PB_TC=2 -- same mapping as train_L1_PC.sh)
#
# Verify a finished ckpt plays BOTH seats by evaluating it twice:
#   online_test_.py --policy-dir <ckpt> --player-id 0   (and 1) \
#     --pool-seed <s> --rule-based-opponents <n> --train-pool-size <n> --eval-pool-size <n> \
#     --recipe-type cross --latent-training --separate-history --self-obs-mode \
#     --use-meta-episode --history-size 5 --include-current-episode --merge-encoder-computation
# The 2026-07-29 FC both-sides run (p effectively 0.0 -- predates the 07-30 p fix)
# verified at LEFT 0.911 / RIGHT 1.000 on the 3 FC L0 partners.
#
# Usage:
#   DOMAIN=FC P=0.25 SEEDS=1 GPUS=2 bash .../train_L1_bothsides.sh
#   DOMAIN=PC P=0.25 TYPES="LO_PB" SEEDS=1 GPUS=2 bash .../train_L1_bothsides.sh
# Env: DOMAIN (FC|PC), P, SEEDS, TYPES (PC only), GPUS, MAXJOBS, EXP_TAG,
#      NUM_ENV_STEPS, ENV_CONFIG, CONSOLIDATE.

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=${WANDB_MODE:-offline}
set -uo pipefail
REPO=${PACE_ROOT}
cd "${REPO}"

DOMAIN=${DOMAIN:-FC}
P=${P:-0.25}
SEEDS=${SEEDS:-"1"}
GPUS=${GPUS:-"2"}
MAXJOBS=${MAXJOBS:-2}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-30000000}
ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml}
CONSOLIDATE=${CONSOLIDATE:-0}

declare -A POOL_SEED=( [LO_PB]=3 [LO_TC]=1 [PB_TC]=2 )

case "${DOMAIN}" in
  FC) LR=3e-4; NPOOL=3; TYPES=${TYPES:-"FC"};;
  PC) LR=1e-3; NPOOL=2; TYPES=${TYPES:-"LO_PB LO_TC PB_TC"};;
  # FI (6-recipe disjoint), added 2026-10-20: same flags as train_L1_FI_p05.sh (lr 1e-3, 2 rule-based
  # L0s per type, --recipe-domain 6recipe --l0-pair-type) plus --both-sides. The FI pool used H=80
  # (fc_small_6recipe_h80.yaml); pass ENV_CONFIG explicitly. User decision: 2 types x 5 seeds first.
  FI) LR=1e-3; NPOOL=2; TYPES=${TYPES:-"R1R2 R3R4"}
      [[ "${ENV_CONFIG}" == "./environment/overcooked/config/fc_small_test.yaml" ]] && \
        ENV_CONFIG=./environment/overcooked/config/fc_small_6recipe_h80.yaml;;
  *)  echo "DOMAIN must be FC, PC or FI"; exit 1;;
esac
EXP_TAG=${EXP_TAG:-L1_${DOMAIN}_bothsides_p$(echo ${P} | tr -d '.')}

read -ra SEED_ARR <<< "${SEEDS}"
read -ra TYPE_ARR <<< "${TYPES}"
read -ra GPU_ARR  <<< "${GPUS}"
LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_TAG}
mkdir -p "${LOG_ROOT}"

echo "=========================================================================="
echo "L1 BOTH-SIDES  domain=${DOMAIN}  p=${P}  lr=${LR}  pool=${NPOOL} L0s"
echo "  Tag:     ${EXP_TAG}"
echo "  Types:   ${TYPES}    Seeds: ${SEEDS}    GPUs: ${GPUS}  MAXJOBS=${MAXJOBS}"
echo "  Env cfg: ${ENV_CONFIG}    steps: ${NUM_ENV_STEPS}"
echo "  Started: $(date '+%F %T')"
echo "=========================================================================="

run_one() {
    local agent_type=$1 seed=$2 gpu_id=$3
    local pool_seed=1
    [[ "${DOMAIN}" == "PC" ]] && pool_seed=${POOL_SEED[${agent_type}]}
    local sub="seed${seed}"; [[ "${DOMAIN}" == "PC" || "${DOMAIN}" == "FI" ]] && sub="${agent_type}/seed${seed}"
    local exp_dir="june_new_exp/${EXP_TAG}/${sub}"
    local log_file="${LOG_ROOT}/$([[ ${DOMAIN} == PC || ${DOMAIN} == FI ]] && echo "${agent_type}_")seed${seed}.log"
    local numa_node; if [[ "${gpu_id}" -ge 4 ]]; then numa_node=1; else numa_node=0; fi
    local domain_flags=""; [[ "${DOMAIN}" == "FI" ]] && domain_flags="--recipe-domain 6recipe --l0-pair-type ${agent_type}"
    local pool_flags="--pool-seed ${pool_seed}"; [[ "${DOMAIN}" == "FI" ]] && pool_flags=""   # FI launcher never passed --pool-seed

    echo "[$(date '+%T')] START ${DOMAIN} ${agent_type} seed${seed} (GPU=${gpu_id} pool-seed=${pool_seed} p=${P})"
    CUDA_VISIBLE_DEVICES=${gpu_id} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config "${ENV_CONFIG}" \
      ${domain_flags} \
      --both-sides \
      --player-id 0 --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr ${LR} --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps ${NUM_ENV_STEPS} \
      --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
      --exp-name "${exp_dir}" \
      --latent-training --e2e-obj \
      --deterministic-latent \
      --use-latent-critic \
      --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
      --latent-dim 128 --kl-coef 0 \
      --hidden-dims 128 128 --act-func relu \
      --train-pool-size ${NPOOL} --eval-pool-size ${NPOOL} \
      --rule-based-opponents ${NPOOL} \
      ${pool_flags} \
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
    echo "[$(date '+%T')] EXIT ${agent_type} seed${seed} rc=${rc} (judge by ckpt, not rc)"
}

gi=0; active=0
for t in "${TYPE_ARR[@]}"; do for s in "${SEED_ARR[@]}"; do
    run_one "$t" "$s" "${GPU_ARR[$((gi % ${#GPU_ARR[@]}))]}" &
    gi=$((gi+1)); active=$((active+1))
    if [[ ${active} -ge ${MAXJOBS} ]]; then wait -n; active=$((active-1)); fi
    sleep 4
done; done
wait
echo "[$(date '+%T')] all both-sides runs terminal"
