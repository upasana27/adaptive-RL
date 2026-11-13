#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# FC both-sides agents vs the 3 rule-based L0 partners (pool-seed 1, p=0.25, as in training): each method's L1 seeds 1-5 and
# its pilot L2, both ego seats, 10x5 protocol. Same flags as eval_L1vsL1_bothsides_FC.sh except the partner pool.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline OMP_NUM_THREADS=1
cd ${PACE_ROOT}
METHODS=${METHODS:-"Ours Generalist LiAM LiAMx LiLi"}; L1_SEEDS=${L1_SEEDS:-"1 2 3 4 5"}; SEATS=${SEATS:-"0 1"}
TEST_PASS=${TEST_PASS:-10}; TEST_EPISODES=${TEST_EPISODES:-5}; GPUS=${GPUS:-"4 5 6 7"}; MAXJOBS=${MAXJOBS:-8}
OUT=${OUT:-logs/Overcooked/june_new_exp/eval_bothsides_FC_vs_L0}; mkdir -p $OUT
declare -A L2
L2[Ours]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_p025/seed1_seed1/ppo/60000000.pt
L2[Generalist]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/Generalist/seed1_seed1/ppo/individual_latest.pt
L2[LiAM]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/LiAM/seed1_seed1/ppo/individual_latest.pt
L2[LiAMx]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/LiAMx/seed1_seed1/ppo/individual_latest.pt
L2[LiLi]=logs/Overcooked/ppo_june_new_exp/L2_FC_bothsides_baselines_p025/bothsides/LiLi/seed1_seed1/ppo/latest.pt
l1ckpt(){ local m=$1 s=$2
  if [ "$m" = Ours ]; then for c in logs/Overcooked/june_new_exp/L1_FC_bothsides_p025/seed$s/ppo/30000000.pt logs/Overcooked/ppo_june_new_exp/L1_FC_bothsides_p025/seed${s}_seed$s/ppo/30000000.pt; do [ -f $c ] && { echo $c; return; }; done
  else for c in logs/Overcooked/june_new_exp/baselines_FC_bothsides_p025/bothsides/$m/seed$s/ppo/30000000.pt logs/Overcooked/ppo_june_new_exp/baselines_FC_bothsides_p025/bothsides/$m/seed${s}_seed$s/ppo/30000000.pt; do [ -f $c ] && { echo $c; return; }; done; fi; }
run_one(){ local m=$1 name=$2 ego=$3 seat=$4 gpu=$5; [ -f "$ego" ] || { echo "[skip] $name missing $ego"; return; }
  local extra=""; case $m in LiAM|LiAMx) extra="--all-has-last-action";; esac
  CUDA_VISIBLE_DEVICES=$gpu python online_test_.py --env-name Overcooked --env_config ./environment/overcooked/config/fc_small_test.yaml --recipe-type cross \
    --algo ppo --exp-name june_new_exp/eval_bothsides_FC_vs_L0/$name --log-dir $OUT/ld_$name --player-id $seat --policy-dir $ego --seed 1 \
    --history-size 5 --include-current-episode --merge-encoder-computation --use-meta-episode --self-obs-mode \
    --train-pool-size 3 --eval-pool-size 0 --rule-based-opponents 3 --pool-seed 1 --p 0.25 --use-train-pool \
    --test-episodes $TEST_EPISODES --test-pass $TEST_PASS $extra > $OUT/$name.log 2>&1
  echo "[done] $name rc=$? $(grep -o "'success_rate': ([0-9.]*" $OUT/$name.log | tail -1 | sed "s/.*(//" | xargs printf 'success=%.3f' 2>/dev/null)"; }
read -ra G <<< "$GPUS"; i=0; active=0
for m in $METHODS; do for seat in $SEATS; do
  run_one $m ${m}_L2_seat$seat "${L2[$m]}" $seat ${G[$((i % ${#G[@]}))]} & i=$((i+1)); active=$((active+1)); [ $active -ge $MAXJOBS ] && { wait -n; active=$((active-1)); }; sleep 2
  for s in $L1_SEEDS; do run_one $m ${m}_L1s${s}_seat$seat "$(l1ckpt $m $s)" $seat ${G[$((i % ${#G[@]}))]} & i=$((i+1)); active=$((active+1)); [ $active -ge $MAXJOBS ] && { wait -n; active=$((active-1)); }; sleep 2; done
done; done; wait; echo "=== all done $(date '+%F %T')"
