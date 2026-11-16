#!/bin/bash
# Level-1 training in Overcooked: an agent of any method trained against the rule-based level-0 partners.
# Called by scripts/overcooked/{FC,PC,FI}/{left,right,bothsides}/train_L1.sh and train_baselines.sh.
#
# Required : DOMAIN (FC | PC | FI)   SIDE (left | right | bothsides)
# Optional : METHODS  "Ours" (train_L1.sh) or "PACE LiAM LiAMx LiLi Generalist" (train_baselines.sh)
#            SEEDS | N_SEEDS (per pool, per type in PC/FI: 1-10 level-2 training partners, 11-15 during-training eval, 16-25 held out),
#            TYPES (PC: LO_PB LO_TC PB_TC; FI: R1R2 R3R4 R5R6 or ALL), P (level-0 noise, default 0.25),
#            GPUS, MAXJOBS, NUM_ENV_STEPS (30M), LR, ENV_CONFIG, EXTRA_ARGS, SMOKE
# Output   : checkpoints logs/Overcooked/ppo_{DOMAIN}/{SIDE}/L1/{METHOD}[/{TYPE}]/seed{N}_seed{N}/ppo/
#            logs        logs/Overcooked/{DOMAIN}/{SIDE}/L1/{METHOD}/[{TYPE}_]seed{N}.log
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${DOMAIN:?set DOMAIN}" "${SIDE:?set SIDE}"
METHODS=${METHODS:-Ours}; SEEDS=${SEEDS:-"1 2 3 4 5"}; GPUS=${GPUS:-0}; MAXJOBS=${MAXJOBS:-4}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-30000000}; SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}; EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
case "${DOMAIN}" in
    FC) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml};        P=${P:-0.25}; TYPES=${TYPES:-FC};                NPOOL=3 ;;
    PC) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml};        P=${P:-0.25}; TYPES=${TYPES:-"LO_PB LO_TC PB_TC"}; NPOOL=2 ;;
    FI) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_6recipe_h80.yaml}; P=${P:-0.25}; TYPES=${TYPES:-"R1R2 R3R4 R5R6"};   NPOOL=2 ;;
    *) echo "DOMAIN must be FC, PC or FI" >&2; exit 1 ;;
esac
# Level-0 pool selection per type. PC pool seeds: LO_PB=3, LO_TC=1, PB_TC=2 (generate_policy_pool order).
type_flags() {
    case "${DOMAIN}" in
        FC) echo "--pool-seed ${POOL_SEED:-1}" ;;
        PC) case "$1" in LO_PB) echo "--pool-seed 3" ;; LO_TC) echo "--pool-seed 1" ;; PB_TC) echo "--pool-seed 2" ;;
                         *) echo "PC TYPES must be LO_PB, LO_TC or PB_TC" >&2; exit 1 ;; esac ;;
        FI) echo "--recipe-domain 6recipe --l0-pair-type $1" ;;
    esac
}
echo "=== Overcooked level 1 | DOMAIN=${DOMAIN} SIDE=${SIDE} METHODS='${METHODS}' TYPES='${TYPES}' SEEDS='$(expand_seeds)' P=${P} steps=${NUM_ENV_STEPS} ==="
for method in ${METHODS}; do
    lr=$(default_lr "${method}" L1 "${DOMAIN}"); [[ -n "${LR:-}" ]] && lr=${LR}
    for type in ${TYPES}; do
        npool=${NPOOL}; [[ "${type}" == ALL ]] && npool=6
        tpath=""; tlog=""; if [[ "${DOMAIN}" != FC ]]; then tpath="/${type}"; tlog="${type}_"; fi
        for seed in $(expand_seeds); do
            throttle; gpu=$(next_gpu)
            exp="${EXP_PREFIX}${DOMAIN}/${SIDE}/L1/${method}${tpath}/seed${seed}"
            logdir="logs/Overcooked/${EXP_PREFIX}${DOMAIN}/${SIDE}/L1/${method}"; mkdir -p "${logdir}"
            log="${logdir}/${tlog}seed${seed}.log"
            echo "[$(date '+%F %T')] start ${exp} (GPU ${gpu}) -> ${log}"
            CUDA_VISIBLE_DEVICES=${gpu} python train_.py --env-name Overcooked --env_config "${ENV_CONFIG}" \
                $(side_flags "${SIDE}") --recipe-type cross \
                --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
                --lr ${lr} --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 \
                --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 15.0 \
                --num-env-steps ${NUM_ENV_STEPS} --save-interval ${SAVE_INTERVAL} --log-interval 10000 \
                --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
                --exp-name "${exp}" \
                --train-pool-size ${npool} --eval-pool-size ${npool} --rule-based-opponents ${npool} $(type_flags "${type}") \
                --seed ${seed} --p ${P} $(method_flags "${method}") ${EXTRA_ARGS:-} \
                ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log}" 2>&1 &
            sleep "${STAGGER_SEC:-5}"
        done
    done
done
wait
echo "[$(date '+%F %T')] all level-1 runs finished"
