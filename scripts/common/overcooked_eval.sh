#!/bin/bash
# Evaluation in Overcooked with the training protocol: TEST_PASS rounds of TEST_EPISODES episodes with the same
# partner, the ego's history cleared between rounds (default 10 x 5; the history period equals TEST_EPISODES).
# Called by scripts/overcooked/{FC,PC,FI}/{left,right,bothsides}/eval_method_vs_L0.sh (MODE=L0) and
# eval_method_vs_L1.sh (MODE=L1).
#
# Required : DOMAIN (FC | PC | FI)   SIDE (left | right | bothsides)   MODE (L0 | L1)
# Ego      : METHOD (Ours), LEVEL (L2 | L1), SEEDS | N_SEEDS, TYPES (FI/PC level-1 egos: one run per type),
#            EGO_BASE (default logs/Overcooked/ppo_{DOMAIN}/{SIDE}/{LEVEL}/{METHOD}), EGO_CKPT (60000000.pt for L2, 30000000.pt for L1)
# Partners : MODE=L0  the rule-based level-0 pool: FC/PC all three recipes (POOL_SEED 1, P); FI the recipes of
#                     L0_TYPE (default: the ego's type for a level-1 ego, ALL for a level-2 ego), P
#            MODE=L1  held-out level-1 partners of PARTNER_METHOD (default METHOD) on the partner side:
#                     FC PARTNER_SEEDS "16 ... 25"; PC/FI PARTNER_SPEC "TYPE:seeds;..." (default seeds 16-25 per type);
#                     PARTNER_BASE overrides the pool directory, PARTNER_PATHS a comma-separated list of checkpoints;
#                     FEED_HISTORY (1) writes the partners' interaction history as in their training
# Other    : TEST_PASS (10), TEST_EPISODES (5), EVAL_SEED (1), GPUS, MAXJOBS, ENV_CONFIG, EXTRA_ARGS, SMOKE
# Output   : logs/Overcooked/{DOMAIN}/{SIDE}/eval_vs_{MODE}/{METHOD}_{LEVEL}[_{TYPE}]_seed{N}_seat{s}.log
#            The success rate is printed per run at the end (from the "Final result" line of each log).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${DOMAIN:?set DOMAIN}" "${SIDE:?set SIDE}" "${MODE:?set MODE (L0 or L1)}"
METHOD=${METHOD:-Ours}; LEVEL=${LEVEL:-L2}; SEEDS=${SEEDS:-"1 2 3 4 5"}; GPUS=${GPUS:-0}; MAXJOBS=${MAXJOBS:-4}
TEST_PASS=${TEST_PASS:-10}; TEST_EPISODES=${TEST_EPISODES:-5}; EVAL_SEED=${EVAL_SEED:-1}
PARTNER_METHOD=${PARTNER_METHOD:-${METHOD}}; PSIDE=$(partner_side "${SIDE}"); FEED_HISTORY=${FEED_HISTORY:-1}
case "${DOMAIN}" in
    FC) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml};        P=${P:-0.25}; TYPES=${TYPES:--} ;;
    PC) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_test.yaml};        P=${P:-0.25}
        if [[ "${LEVEL}" == L1 ]]; then TYPES=${TYPES:-"LO_PB LO_TC PB_TC"}; else TYPES=${TYPES:--}; fi ;;
    FI) ENV_CONFIG=${ENV_CONFIG:-./environment/overcooked/config/fc_small_6recipe_h80.yaml}; P=${P:-0.25}
        if [[ "${LEVEL}" == L1 ]]; then TYPES=${TYPES:-"R1R2 R3R4 R5R6"}; else TYPES=${TYPES:--}; fi ;;
    *) echo "DOMAIN must be FC, PC or FI" >&2; exit 1 ;;
esac
if [[ "${LEVEL}" == L2 ]]; then EGO_CKPT=${EGO_CKPT:-60000000.pt}; else EGO_CKPT=${EGO_CKPT:-30000000.pt}; fi
EGO_BASE=${EGO_BASE:-logs/Overcooked/ppo_${EXP_PREFIX}${DOMAIN}/${SIDE}/${LEVEL}/${METHOD}}
OUT=${OUT:-logs/Overcooked/${EXP_PREFIX}${DOMAIN}/${SIDE}/eval_vs_${MODE}}; mkdir -p "${OUT}"

# Partner flags.
partner_flags() {  # $1 ego type ('-' for none)
    if [[ "${MODE}" == L0 ]]; then
        case "${DOMAIN}" in
            FC|PC) echo "--recipe-type cross --train-pool-size 3 --eval-pool-size 0 --rule-based-opponents 3 --pool-seed ${POOL_SEED:-1} --p ${P} --use-train-pool" ;;
            FI)    local t=${L0_TYPE:-$1}; [[ "${t}" == - ]] && t=ALL
                   echo "--l0-pair-type ${t} --p ${P} --latent-training --separate-history" ;;
        esac
    else
        local paths=${PARTNER_PATHS:-}
        if [[ -z "${paths}" ]]; then
            local base=${PARTNER_BASE:-logs/Overcooked/ppo_${EXP_PREFIX}${DOMAIN}/${PSIDE}/L1/${PARTNER_METHOD}} p
            if [[ "${DOMAIN}" == FC ]]; then
                for s in ${PARTNER_SEEDS:-16 17 18 19 20 21 22 23 24 25}; do
                    p=$(resolve_ckpt "${base}" "${s}" 30000000.pt) || { echo "missing partner ${p}" >&2; exit 1; }
                    paths="${paths:+${paths},}${p}"
                done
            else
                local spec=${PARTNER_SPEC:-}
                if [[ -z "${spec}" ]]; then
                    if [[ "${DOMAIN}" == PC ]]; then spec="LO_PB:16,17,18,19,20,21,22,23,24,25;LO_TC:16,17,18,19,20,21,22,23,24,25;PB_TC:16,17,18,19,20,21,22,23,24,25"; else spec="R1R2:16,17,18,19,20,21,22,23,24,25;R3R4:16,17,18,19,20,21,22,23,24,25;R5R6:16,17,18,19,20,21,22,23,24,25"; fi
                fi
                for ts in ${spec//;/ }; do
                    local t=${ts%%:*}
                    for s in ${ts#*:}; do for s in ${s//,/ }; do
                        p=$(resolve_ckpt "${base}/${t}" "${s}" 30000000.pt) || { echo "missing partner ${p}" >&2; exit 1; }
                        paths="${paths:+${paths},}${p}"
                    done; done
                done
            fi
        fi
        local n; n=$(echo "${paths}" | tr ',' '\n' | grep -c .)
        echo "--train-pool-size ${n} --partner-policy-paths ${paths} $([[ "${FEED_HISTORY}" == 1 ]] && echo --feed-partner-history)"
    fi
}
echo "=== Overcooked eval vs ${MODE} | DOMAIN=${DOMAIN} SIDE=${SIDE} ego=${METHOD} ${LEVEL} seeds='$(expand_seeds)' types='${TYPES}' protocol ${TEST_PASS}x${TEST_EPISODES} ==="
logs=()
for type in ${TYPES}; do
    tpath=""; tname=""; if [[ "${type}" != - ]]; then tpath="/${type}"; tname="_${type}"; fi
    pflags=$(partner_flags "${type}")
    for seed in $(expand_seeds); do
        ego=$(resolve_ckpt "${EGO_BASE}${tpath}" "${seed}" "${EGO_CKPT}") || { echo "missing ego checkpoint ${ego}" >&2; exit 1; }
        for seat in $(seats_for_side "${SIDE}"); do
            throttle; gpu=$(next_gpu)
            name="${METHOD}_${LEVEL}${tname}_seed${seed}_seat${seat}"; log="${OUT}/${name}.log"; logs+=("${log}")
            echo "[$(date '+%F %T')] ${name}: ego ${ego} (GPU ${gpu})"
            CUDA_VISIBLE_DEVICES=${gpu} python online_test_.py --env-name Overcooked --env_config "${ENV_CONFIG}" --algo ppo \
                --exp-name "${EXP_PREFIX}${DOMAIN}/${SIDE}/eval_vs_${MODE}/${name}" --log-dir "${OUT}/ld_${name}" \
                --player-id ${seat} --policy-dir "${ego}" --seed ${EVAL_SEED} \
                --history-size ${TEST_EPISODES} --include-current-episode --merge-encoder-computation --use-meta-episode --self-obs-mode \
                --test-pass ${TEST_PASS} --test-episodes ${TEST_EPISODES} \
                ${pflags} $(ego_eval_flags "${METHOD}") ${EXTRA_ARGS:-} > "${log}" 2>&1 &
            sleep 3
        done
    done
done
wait
echo "=== results (success rate per run) ==="
for log in "${logs[@]}"; do
    res=$(grep -E "^Final result" "${log}" | tail -1 | grep -oE "'success_rate': \([^,]*" | grep -oE "[0-9.]+$")
    printf '%-60s success %s\n' "$(basename "${log}" .log)" "${res:-<no result, see ${log}>}"
done
