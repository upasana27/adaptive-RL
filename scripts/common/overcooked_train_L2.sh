#!/bin/bash
# Level-2 training in Overcooked: an agent of any method trained against the level-1 pool OF THE SAME
# METHOD (nested regime). Called by scripts/overcooked/{FC,PC,FI}/{left,right,bothsides}/train_L2.sh.
#
# Required : DOMAIN (FC | PC | FI)   SIDE (left | right | bothsides)
# Optional : METHOD (Ours), SEEDS | N_SEEDS (1..5), GPUS, MAXJOBS, NUM_ENV_STEPS (60M), LR, NUM_PROCS,
#            L1_BASE_DIR   where the partner pool is (default: this method's level-1 runs on the partner side)
#            FC : TRAIN_SEEDS "1,2,3,4,5,6,7,8,9,10"  EVAL_SEEDS "11,12,13,14,15" (during-training eval; held-out 16-25 are for eval_method_vs_L1.sh)  L1_CKPT (30000000.pt)
#            PC : TRAIN_SPEC  "LO_PB:1,2,3,4,5,6,7,8,9,10;..." (10 per type)  EVAL_SPEC "LO_PB:11,12,13,14,15;..." (5 per type)
#            FI : TRAIN_SPEC  "R1R2:1,2,3,4,5,6,7,8,9,10;..." (10 per type)  EVAL_SPEC "R1R2:11,12,13,14,15;..." (5 per type)
#            FEED_HISTORY (1; 0 for PC) writes the partners' interaction history during training,
#            ENV_CONFIG, EXTRA_ARGS, SMOKE
# The partner side is the opposite seat for left/right and the both-sides pool for bothsides.
# Output   : checkpoints logs/Overcooked/ppo_{DOMAIN}/{SIDE}/L2/{METHOD}/seed{N}_seed{N}/ppo/
#            logs        logs/Overcooked/{DOMAIN}/{SIDE}/L2/{METHOD}/seed{N}.log
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${DOMAIN:?set DOMAIN}" "${SIDE:?set SIDE}"
METHOD=${METHOD:-Ours}; SEEDS=${SEEDS:-"1 2 3 4 5"}; GPUS=${GPUS:-0}; MAXJOBS=${MAXJOBS:-3}
NUM_ENV_STEPS=${NUM_ENV_STEPS:-60000000}; SAVE_INTERVAL=${SAVE_INTERVAL:-5000000}; EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
PSIDE=$(partner_side "${SIDE}")
L1_BASE_DIR=${L1_BASE_DIR:-logs/Overcooked/ppo_${EXP_PREFIX}${DOMAIN}/${PSIDE}/L1/${METHOD}}
case "${DOMAIN}" in
    FC) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml}
        TRAIN_SEEDS=${TRAIN_SEEDS:-"1,2,3,4,5,6,7,8,9,10"}; EVAL_SEEDS=${EVAL_SEEDS:-"11,12,13,14,15"}; FEED_HISTORY=${FEED_HISTORY:-1}
        ntrain=$(spec_count "${TRAIN_SEEDS}"); neval=$(spec_count "${EVAL_SEEDS}")
        pool_flags="--use-l1-partners --l1-partner-base-dir ${L1_BASE_DIR} --l1-partner-ckpt ${L1_CKPT:-30000000.pt} \
--l1-train-seeds ${TRAIN_SEEDS} --l1-eval-seeds ${EVAL_SEEDS}" ;;
    PC) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml}
        TRAIN_SPEC=${TRAIN_SPEC:-"LO_PB:1,2,3,4,5,6,7,8,9,10;LO_TC:1,2,3,4,5,6,7,8,9,10;PB_TC:1,2,3,4,5,6,7,8,9,10"}; EVAL_SPEC=${EVAL_SPEC:-"LO_PB:11,12,13,14,15;LO_TC:11,12,13,14,15;PB_TC:11,12,13,14,15"}
        FEED_HISTORY=${FEED_HISTORY:-0}
        ntrain=$(spec_count "${TRAIN_SPEC}"); neval=$(spec_count "${EVAL_SPEC}")
        pool_flags="--use-l1-specialized --l1-specialized-base-dir ${L1_BASE_DIR} \
--l1-specialized-train ${TRAIN_SPEC} --l1-specialized-eval ${EVAL_SPEC}" ;;
    FI) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_6recipe_h80_b3c3_pair_c80_plate5.yaml}
        TRAIN_SPEC=${TRAIN_SPEC:-"R1R2:1,2,3,4,5,6,7,8,9,10;R3R4:1,2,3,4,5,6,7,8,9,10;R5R6:1,2,3,4,5,6,7,8,9,10"}; EVAL_SPEC=${EVAL_SPEC:-"R1R2:11,12,13,14,15;R3R4:11,12,13,14,15;R5R6:11,12,13,14,15"}
        FEED_HISTORY=${FEED_HISTORY:-1}
        ntrain=$(spec_count "${TRAIN_SPEC}"); neval=$(spec_count "${EVAL_SPEC}")
        pool_flags="--recipe-domain 6recipe --use-l1-disjoint --l1-disjoint-base-dir ${L1_BASE_DIR} \
--l1-disjoint-train ${TRAIN_SPEC} --l1-disjoint-eval ${EVAL_SPEC}" ;;
    *) echo "DOMAIN must be FC, PC or FI" >&2; exit 1 ;;
esac
[[ "${FEED_HISTORY}" == 1 ]] && pool_flags="${pool_flags} --feed-partner-history"
lr=$(default_lr "${METHOD}" L2 "${DOMAIN}"); [[ -n "${LR:-}" ]] && lr=${LR}
# One environment process per training partner at least (train_.py requires num-processes >= pool size).
if [[ "${METHOD}" == Ours ]]; then nproc=${NUM_PROCS:-15}; else nproc=${NUM_PROCS:-12}; fi
(( nproc < ntrain )) && nproc=${ntrain}
[[ -d "${L1_BASE_DIR}" ]] || { echo "partner pool not found: ${L1_BASE_DIR} (set L1_BASE_DIR)" >&2; exit 1; }
echo "=== Overcooked level 2 | DOMAIN=${DOMAIN} SIDE=${SIDE} METHOD=${METHOD} SEEDS='$(expand_seeds)' partners=${L1_BASE_DIR} (${ntrain} train / ${neval} eval) steps=${NUM_ENV_STEPS} ==="
prev_log=""; prev_pid=""
for seed in $(expand_seeds); do
    throttle; gpu=$(next_gpu)
    [[ -n "${prev_log}" ]] && stagger_wait "${prev_log}" "${prev_pid}"
    exp="${EXP_PREFIX}${DOMAIN}/${SIDE}/L2/${METHOD}/seed${seed}"
    logdir="logs/Overcooked/${EXP_PREFIX}${DOMAIN}/${SIDE}/L2/${METHOD}"; mkdir -p "${logdir}"; log="${logdir}/seed${seed}.log"
    echo "[$(date '+%F %T')] start ${exp} (GPU ${gpu}) -> ${log}"
    CUDA_VISIBLE_DEVICES=${gpu} python train_.py --env-name Overcooked --env_config "${ENV_CONFIG}" \
        $(side_flags "${SIDE}") --recipe-type cross \
        --algo ppo --num-steps 1000 --num-processes ${nproc} --use-gae --eps 1e-8 --use-proper-time-limit \
        --lr ${lr} --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 \
        --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 15.0 \
        --num-env-steps ${NUM_ENV_STEPS} --save-interval ${SAVE_INTERVAL} --log-interval 10000 \
        --eval-interval ${EVAL_INTERVAL} --eval-episodes 5 \
        --exp-name "${exp}" \
        --train-pool-size ${ntrain} --eval-pool-size ${neval} --rule-based-opponents 0 ${pool_flags} \
        --seed ${seed} --p 0.0 $(method_flags "${METHOD}") ${EXTRA_ARGS:-} \
        ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > "${log}" 2>&1 &
    prev_pid=$!; prev_log=${log}
done
wait
echo "[$(date '+%F %T')] all level-2 runs finished"
