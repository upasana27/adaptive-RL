#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# "Without regime" row of the FC both-sides comparison: each method's both-sides L1 (trained vs L0 only), seeds 1-5 as EGO,
# vs that method's HELD-OUT both-sides L1 seeds 6-10 as partners, both seats (ego player 0 and player 1), 10x5 protocol.
# Ours = L1_FC_bothsides_p025 (latent arch, no peer-id head). Baselines = baselines_FC_bothsides_p025/bothsides/{M}.
# LiAM/LiAMx egos need --all-has-last-action (their obs carries their own last action). Partners get history fed.
# Usage: METHODS="Ours LiAM LiAMx LiLi Generalist" EGO_SEEDS="1 2 3 4 5" PARTNER_SEEDS="6 7 8 9 10" SEATS="0 1" \
#        TEST_PASS=10 TEST_EPISODES=5 GPUS="4 5 6 7" MAXJOBS=6 bash .../eval_L1vsL1_bothsides_FC.sh
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-pace}"
export WANDB_MODE=offline OMP_NUM_THREADS=1
cd ${PACE_ROOT}
METHODS=${METHODS:-"Ours LiAM LiAMx LiLi Generalist"}; EGO_SEEDS=${EGO_SEEDS:-"1 2 3 4 5"}; PARTNER_SEEDS=${PARTNER_SEEDS:-"6 7 8 9 10"}
SEATS=${SEATS:-"0 1"}; TEST_PASS=${TEST_PASS:-10}; TEST_EPISODES=${TEST_EPISODES:-5}; GPUS=${GPUS:-"4 5 6 7"}; MAXJOBS=${MAXJOBS:-6}
OUT=${OUT:-logs/Overcooked/june_new_exp/eval_L1vsL1_bothsides_FC}; mkdir -p $OUT
ckpt(){ # method seed -> path
  local m=$1 s=$2
  if [ "$m" = Ours ]; then for c in logs/Overcooked/june_new_exp/L1_FC_bothsides_p025/seed$s/ppo/30000000.pt logs/Overcooked/ppo_june_new_exp/L1_FC_bothsides_p025/seed${s}_seed$s/ppo/30000000.pt; do [ -f $c ] && { echo $c; return; }; done
  else for c in logs/Overcooked/june_new_exp/baselines_FC_bothsides_p025/bothsides/$m/seed$s/ppo/30000000.pt logs/Overcooked/ppo_june_new_exp/baselines_FC_bothsides_p025/bothsides/$m/seed${s}_seed$s/ppo/30000000.pt; do [ -f $c ] && { echo $c; return; }; done; fi
}
run_one(){ local m=$1 e=$2 seat=$3 gpu=$4
  local ego; ego=$(ckpt $m $e); [ -n "$ego" ] || { echo "[skip] $m ego seed$e missing"; return; }
  local P=""; for s in $PARTNER_SEEDS; do local c; c=$(ckpt $m $s); [ -n "$c" ] || { echo "[skip] $m partner seed$s missing"; return; }; P="$P,$c"; done; P=${P#,}
  local extra=""; case $m in LiAM|LiAMx) extra="--all-has-last-action";; esac
  local name=${m}_ego${e}_seat${seat}
  CUDA_VISIBLE_DEVICES=$gpu python online_test_.py --env-name Overcooked --env_config ./environment/overcooked/config/fc_small_test.yaml \
    --algo ppo --exp-name june_new_exp/eval_L1vsL1_bothsides_FC/$name --log-dir $OUT/ld_$name --player-id $seat --policy-dir $ego --seed 1 \
    --history-size 5 --include-current-episode --merge-encoder-computation --use-meta-episode --self-obs-mode \
    --train-pool-size $(echo $PARTNER_SEEDS | wc -w) --test-episodes $TEST_EPISODES --test-pass $TEST_PASS \
    --feed-partner-history --partner-policy-paths "$P" $extra > $OUT/$name.log 2>&1
  echo "[done] $name rc=$? $(grep -o "'success_rate': ([0-9.]*" $OUT/$name.log | tail -1 | sed "s/.*(//" | xargs printf 'success=%.3f' 2>/dev/null)"
}
read -ra G <<< "$GPUS"; i=0; active=0
for m in $METHODS; do for e in $EGO_SEEDS; do for seat in $SEATS; do
  run_one $m $e $seat ${G[$((i % ${#G[@]}))]} & i=$((i+1)); active=$((active+1))
  if [ $active -ge $MAXJOBS ]; then wait -n; active=$((active-1)); fi; sleep 2
done; done; done; wait
echo "=== all done $(date '+%F %T')"
