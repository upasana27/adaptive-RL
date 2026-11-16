#!/bin/bash
# Level-1 training in Level-Based Foraging: an agent of any method against the three rule-based level-0
# partners (one per item type). Called by scripts/lbf/{left,right,bothsides}/train_L1.sh and train_baselines.sh.
#
# Required : SIDE (left | right | bothsides)
# Optional : METHODS (Ours), SEEDS | N_SEEDS (1-10 level-2 training partners, 11-15 during-training eval, 16-25 held out), P (level-0 noise, 0),
#            GPUS, MAXJOBS, NUM_ENV_STEPS (30M), LR, HORIZON (80),
#            L1_EVAL_SPEC "TAG:seeds" adds level-1 partners to the during-training evaluation pool
#            (checkpoints under L1_BASE_DIR, default logs/LBF/ppo_LBF/{partner side}/L1), EXTRA_ARGS, SMOKE
# Output   : checkpoints logs/LBF/ppo_LBF/{SIDE}/L1/{METHOD}/seed{N}_seed{N}/ppo/   logs logs/LBF/LBF/{SIDE}/L1/{METHOD}/seed{N}.log
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${SIDE:?set SIDE}"
METHODS=${METHODS:-Ours}; SEEDS=${SEEDS:-"1 2 3 4 5"}; GPUS=${GPUS:-0}; MAXJOBS=${MAXJOBS:-4}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-30000000}; SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}; EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
HORIZON=${HORIZON:-80}; ENV_CONFIG=${ENV_CONFIG:-./environment/lbf/config/lbf_fc_3type.yaml}
PSIDE=$(partner_side "${SIDE}"); L1_BASE_DIR=${L1_BASE_DIR:-logs/LBF/ppo_${EXP_PREFIX}LBF/${PSIDE}/L1}
eval_flags=""; [[ -n "${L1_EVAL_SPEC:-}" ]] && eval_flags="--lbf-l1-eval-spec ${L1_EVAL_SPEC} --l1-specialized-base-dir ${L1_BASE_DIR}"
echo "=== LBF level 1 | SIDE=${SIDE} METHODS='${METHODS}' SEEDS='$(expand_seeds)' horizon=${HORIZON} steps=${NUM_ENV_STEPS} ==="
for method in ${METHODS}; do
    lr=${LR:-1e-3}
    for seed in $(expand_seeds); do
        throttle; gpu=$(next_gpu)
        exp="${EXP_PREFIX}LBF/${SIDE}/L1/${method}/seed${seed}"
        logdir="logs/LBF/${EXP_PREFIX}LBF/${SIDE}/L1/${method}"; mkdir -p "${logdir}"; log="${logdir}/seed${seed}.log"
        echo "[$(date '+%F %T')] start ${exp} (GPU ${gpu}) -> ${log}"
        CUDA_VISIBLE_DEVICES=${gpu} python train_.py --env-name LBF --env_config "${ENV_CONFIG}" \
            --lbf-num-fruit-types 3 --lbf-horizon ${HORIZON} --horizon ${HORIZON} $(side_flags "${SIDE}") \
            --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
            --lr ${lr} --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 4 \
            --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 15.0 \
            --num-env-steps ${NUM_ENV_STEPS} --save-interval ${SAVE_INTERVAL} --log-interval 10000 \
            --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
            --exp-name "${exp}" \
            --train-pool-size 3 --eval-pool-size 3 --rule-based-opponents 3 ${eval_flags} \
            --seed ${seed} --p ${P:-0.0} $(method_flags "${method}") ${EXTRA_ARGS:-} \
            ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log}" 2>&1 &
        sleep "${STAGGER_SEC:-5}"
    done
done
wait
echo "[$(date '+%F %T')] all LBF level-1 runs finished"
