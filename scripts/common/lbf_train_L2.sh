#!/bin/bash
# Level-2 training in Level-Based Foraging against the level-1 pool of the same method.
# Called by scripts/lbf/{left,right,bothsides}/train_L2.sh.
#
# Required : SIDE (left | right | bothsides)
# Optional : METHOD (Ours), SEEDS | N_SEEDS (1..5), GPUS, MAXJOBS, NUM_ENV_STEPS (60M), LR (3e-4 Ours, 1e-3 baselines),
#            NUM_PROCS (12), HORIZON (80), L1_BASE_DIR (logs/LBF/ppo_LBF/{partner side}/L1),
#            TRAIN_SPEC "{METHOD}:1,2,3,4,5,6,7,8,9,10"  EVAL_SPEC "{METHOD}:11,12,13,14,15"  (TAG:seeds, TAG = directory under L1_BASE_DIR),
#            EXTRA_ARGS, SMOKE
# Output   : checkpoints logs/LBF/ppo_LBF/{SIDE}/L2/{METHOD}/seed{N}_seed{N}/ppo/   logs logs/LBF/LBF/{SIDE}/L2/{METHOD}/seed{N}.log
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${SIDE:?set SIDE}"
METHOD=${METHOD:-Ours}; SEEDS=${SEEDS:-"1 2 3 4 5"}; GPUS=${GPUS:-0}; MAXJOBS=${MAXJOBS:-3}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-60000000}; SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}; EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
HORIZON=${HORIZON:-80}; ENV_CONFIG=${ENV_CONFIG:-./environment/lbf/config/lbf_fc_3type.yaml}
PSIDE=$(partner_side "${SIDE}"); L1_BASE_DIR=${L1_BASE_DIR:-logs/LBF/ppo_${EXP_PREFIX}LBF/${PSIDE}/L1}
TRAIN_SPEC=${TRAIN_SPEC:-"${METHOD}:1,2,3,4,5,6,7,8,9,10"}; EVAL_SPEC=${EVAL_SPEC:-"${METHOD}:11,12,13,14,15"}
ntrain=$(spec_count "${TRAIN_SPEC}"); neval=$(spec_count "${EVAL_SPEC}")
lr=$(default_lr "${METHOD}" L2 LBF); [[ -n "${LR:-}" ]] && lr=${LR}
nproc=${NUM_PROCS:-12}; (( nproc < ntrain )) && nproc=${ntrain}
[[ -d "${L1_BASE_DIR}" ]] || { echo "partner pool not found: ${L1_BASE_DIR} (set L1_BASE_DIR)" >&2; exit 1; }
echo "=== LBF level 2 | SIDE=${SIDE} METHOD=${METHOD} SEEDS='$(expand_seeds)' partners=${L1_BASE_DIR} (${ntrain} train / ${neval} eval) ==="
prev_log=""; prev_pid=""
for seed in $(expand_seeds); do
    throttle; gpu=$(next_gpu)
    [[ -n "${prev_log}" ]] && stagger_wait "${prev_log}" "${prev_pid}"
    exp="${EXP_PREFIX}LBF/${SIDE}/L2/${METHOD}/seed${seed}"
    logdir="logs/LBF/${EXP_PREFIX}LBF/${SIDE}/L2/${METHOD}"; mkdir -p "${logdir}"; log="${logdir}/seed${seed}.log"
    echo "[$(date '+%F %T')] start ${exp} (GPU ${gpu}) -> ${log}"
    CUDA_VISIBLE_DEVICES=${gpu} python train_.py --env-name LBF --env_config "${ENV_CONFIG}" \
        --lbf-num-fruit-types 3 --lbf-horizon ${HORIZON} --horizon ${HORIZON} $(side_flags "${SIDE}") \
        --algo ppo --num-steps 1000 --num-processes ${nproc} --use-gae --eps 1e-8 --use-proper-time-limit \
        --lr ${lr} --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 4 \
        --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 15.0 \
        --num-env-steps ${NUM_ENV_STEPS} --save-interval ${SAVE_INTERVAL} --log-interval 10000 \
        --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
        --exp-name "${exp}" \
        --train-pool-size ${ntrain} --eval-pool-size ${neval} --rule-based-opponents 0 \
        --use-l1-specialized --l1-specialized-base-dir "${L1_BASE_DIR}" \
        --l1-specialized-train "${TRAIN_SPEC}" --l1-specialized-eval "${EVAL_SPEC}" \
        --seed ${seed} --p 0.0 $(method_flags "${METHOD}") ${EXTRA_ARGS:-} \
        ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log}" 2>&1 &
    prev_pid=$!; prev_log=${log}
done
wait
echo "[$(date '+%F %T')] all LBF level-2 runs finished"
