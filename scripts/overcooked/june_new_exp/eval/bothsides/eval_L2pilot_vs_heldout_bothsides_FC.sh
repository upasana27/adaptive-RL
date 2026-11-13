#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# "With regime" on the SAME held-out seeds as eval_L1vsL1_bothsides_FC.sh: each method's both-sides L2 (3/2 pilot design,
# latest checkpoint) as ego, both seats, vs its own held-out both-sides L1 seeds 6-10, 10x5 protocol.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline OMP_NUM_THREADS=1
cd ${PACE_ROOT}
PARTNER_SEEDS=${PARTNER_SEEDS:-"6 7 8 9 10"}; SEATS=${SEATS:-"0 1"}; TEST_PASS=${TEST_PASS:-10}; TEST_EPISODES=${TEST_EPISODES:-5}
GPUS=${GPUS:-"4 5"}; MAXJOBS=${MAXJOBS:-4}; OUT=${OUT:-logs/Overcooked/june_new_exp/eval_L2pilot_vs_heldout_bothsides_FC}; mkdir -p $OUT
declare -A EGO
EGO[Ours]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_p025/seed1_seed1/ppo/60000000.pt
EGO[Generalist]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/Generalist/seed1_seed1/ppo/individual_latest.pt
EGO[LiAM]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/LiAM/seed1_seed1/ppo/individual_latest.pt
EGO[LiAMx]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/LiAMx/seed1_seed1/ppo/individual_latest.pt
EGO[LiLi]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/LiLi/seed1_seed1/ppo/latest.pt
ckpt(){ local m=$1 s=$2
  if [ "$m" = Ours ]; then for c in logs/Overcooked/june_new_exp/L1_FC_bothsides_p025/seed$s/ppo/30000000.pt logs/Overcooked/ppo_june_new_exp/L1_FC_bothsides_p025/seed${s}_seed$s/ppo/30000000.pt; do [ -f $c ] && { echo $c; return; }; done
  else for c in logs/Overcooked/june_new_exp/baselines_FC_bothsides_p025/bothsides/$m/seed$s/ppo/30000000.pt logs/Overcooked/ppo_june_new_exp/baselines_FC_bothsides_p025/bothsides/$m/seed${s}_seed$s/ppo/30000000.pt; do [ -f $c ] && { echo $c; return; }; done; fi; }
run_one(){ local m=$1 seat=$2 gpu=$3; local ego=${EGO[$m]}; [ -f "$ego" ] || { echo "[skip] $m missing $ego"; return; }
  local P=""; for s in $PARTNER_SEEDS; do P="$P,$(ckpt $m $s)"; done; P=${P#,}
  local extra=""; case $m in LiAM|LiAMx) extra="--all-has-last-action";; esac
  local name=${m}_L2_seat${seat}
  CUDA_VISIBLE_DEVICES=$gpu python online_test_.py --env-name Overcooked --env_config ./environment/overcooked/config/fc_small_test.yaml \
    --algo ppo --exp-name june_new_exp/eval_L2pilot_vs_heldout_bothsides_FC/$name --log-dir $OUT/ld_$name --player-id $seat --policy-dir $ego --seed 1 \
    --history-size 5 --include-current-episode --merge-encoder-computation --use-meta-episode --self-obs-mode \
    --train-pool-size $(echo $PARTNER_SEEDS | wc -w) --test-episodes $TEST_EPISODES --test-pass $TEST_PASS \
    --feed-partner-history --partner-policy-paths "$P" $extra > $OUT/$name.log 2>&1
  echo "[done] $name rc=$? $(grep -o "'success_rate': ([0-9.]*" $OUT/$name.log | tail -1 | sed "s/.*(//" | xargs printf 'success=%.3f' 2>/dev/null)"; }
read -ra G <<< "$GPUS"; i=0; active=0
for m in Ours Generalist LiAM LiAMx LiLi; do for seat in $SEATS; do run_one $m $seat ${G[$((i % ${#G[@]}))]} & i=$((i+1)); active=$((active+1)); if [ $active -ge $MAXJOBS ]; then wait -n; active=$((active-1)); fi; sleep 2; done; done; wait
echo "=== all done $(date '+%F %T')"
