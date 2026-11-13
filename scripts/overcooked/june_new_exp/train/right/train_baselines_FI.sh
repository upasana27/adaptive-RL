#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Baseline L1 agents (LiAM, LiAMx, LiLi, Generalist) in the FI domain (6-recipe disjoint, fc_small_6recipe_h80),
# one run per (method, type, seed), vs that type's 2 rule-based L0s. User decision 2026-10-31: 5 seeds per
# method x type, RIGHT seat first, then both-sides.
# Architecture flags are copied verbatim from train_baselines_FC_right.sh::method_flags; the FI regime flags
# (--recipe-domain 6recipe --l0-pair-type, 2 L0s, no --pool-seed, h80 config, p=0.25) from train_L1_FI_p05.sh.
#
# Output: logs/Overcooked/june_new_exp/${EXP_GROUP}/${SIDE}/{METHOD}_{TYPE}_seed{N}.log
#         ckpts logs/Overcooked/ppo_june_new_exp/${EXP_GROUP}/${SIDE}/{METHOD}/{TYPE}/seed{N}_seed{N}/ppo/
#         (consolidate ONCE at the end: consolidate_checkpoints.sh ${EXP_GROUP} ${SIDE})
# Usage: setsid nohup bash scripts/overcooked/june_new_exp/train/right/train_baselines_FI.sh \
#          > logs/Overcooked/june_new_exp/baselines_FI_p025_h80/right/_launcher.log 2>&1 < /dev/null &
# Env: SIDE(right) PLAYER_ID(1) BOTH_SIDES(0) P(0.25) SEEDS("1 2 3 4 5") METHODS TYPES GPUS MAXJOBS(4)
#      NUM_ENV_STEPS SAVE_INTERVAL EVAL_INTERVAL EXP_GROUP(baselines_FI_p025_h80) ENV_CONFIG
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -uo pipefail
REPO=${PACE_ROOT}
cd "${REPO}"
SIDE=${SIDE:-right}
PLAYER_ID=${PLAYER_ID:-1}
BOTH_SIDES=${BOTH_SIDES:-0}
BOTH_SIDES_FLAG=""
if [[ "${BOTH_SIDES}" == "1" ]]; then SIDE=bothsides; PLAYER_ID=0; BOTH_SIDES_FLAG="--both-sides"; fi
P=${P:-0.25}
SEEDS=${SEEDS:-"1 2 3 4 5"}
METHODS=${METHODS:-"LiAM LiAMx LiLi Generalist"}
TYPES=${TYPES:-"R1R2 R3R4 R5R6"}
GPUS=${GPUS:-"4 5 6 7"}
MAXJOBS=${MAXJOBS:-4}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-30000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
EXP_GROUP=${EXP_GROUP:-baselines_FI_p025_h80}
ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_6recipe_h80.yaml}
LOG_ROOT=logs/Overcooked/june_new_exp/${EXP_GROUP}/${SIDE}
mkdir -p "${LOG_ROOT}"
read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
echo "=========================================================================="
echo "FI baselines  group=${EXP_GROUP} side=${SIDE} player=${PLAYER_ID} both_sides=${BOTH_SIDES}"
echo "  Methods: ${METHODS}  Types: ${TYPES}  Seeds: ${SEEDS}  GPUs: ${GPUS}  MAXJOBS=${MAXJOBS}"
echo "  Env: ${ENV_CONFIG}  p=${P}  steps=${NUM_ENV_STEPS}   Started: $(date '+%F %T')"
echo "=========================================================================="
method_flags() {
    case "$1" in
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
        *) echo "UNKNOWN_METHOD_$1" ;;
    esac
}
run_one() {
    local method=$1 agent_type=$2 seed=$3 gpu=$4
    local exp_dir="june_new_exp/${EXP_GROUP}/${SIDE}/${method}/${agent_type}/seed${seed}"
    local log_file="${LOG_ROOT}/${method}_${agent_type}_seed${seed}.log"
    local numa_node=0; [[ "${gpu}" -ge 4 ]] && numa_node=1
    echo "[$(date '+%T')] START ${method}_${agent_type}_seed${seed} (GPU=${gpu})"
    CUDA_VISIBLE_DEVICES=${gpu} numactl --cpunodebind=${numa_node} --membind=${numa_node} \
    python train_.py --env-name Overcooked \
      --env_config "${ENV_CONFIG}" \
      --recipe-domain 6recipe --l0-pair-type ${agent_type} \
      --player-id ${PLAYER_ID} ${BOTH_SIDES_FLAG} --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr 1e-3 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps ${NUM_ENV_STEPS} \
      --save-interval ${SAVE_INTERVAL} --log-interval 10000 --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
      --exp-name "${exp_dir}" \
      --train-pool-size 2 --eval-pool-size 2 \
      --rule-based-opponents 2 \
      --seed ${seed} \
      --p ${P} \
      $(method_flags "${method}") \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log_file}" 2>&1 < /dev/null
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then echo "[$(date '+%T')] DONE  ${method}_${agent_type}_seed${seed} (rc=0)";
    else echo "[$(date '+%T')] FAIL  ${method}_${agent_type}_seed${seed} (rc=${rc}) — see ${log_file}"; fi
}
JOBS=()
for seed in ${SEEDS}; do for method in ${METHODS}; do for agent_type in ${TYPES}; do JOBS+=("${method} ${agent_type} ${seed}"); done; done; done
gpu_idx=0; active=0
for job in "${JOBS[@]}"; do
    read -r jm jt js <<< "${job}"
    gpu=${GPU_ARR[$((gpu_idx % NUM_GPUS))]}
    run_one "${jm}" "${jt}" "${js}" "${gpu}" &
    gpu_idx=$((gpu_idx + 1)); active=$((active + 1))
    if [[ ${active} -ge ${MAXJOBS} ]]; then wait -n; active=$((active - 1)); fi
    sleep 20
done
wait
echo "=== all FI baseline runs finished $(date '+%F %T') (consolidate: bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh ${EXP_GROUP} ${SIDE})"
