#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Evaluate ONE ego (an L2-M or a left-L1-M) against method M's HELD-OUT right L1 seeds.
#
# This is the headline comparison of the per-method regime experiment:
#   L2-M  vs held-out L1-M seeds  (should succeed)
#   L1-M-left vs the SAME seeds   (ablation; should be far worse)
#
# Protocol: 10 rounds x 5-episode meta-episodes (history clears every 5).
# Path resolution: ego + partner checkpoints are searched across consolidated
# (june_new_exp) and raw (ppo_june_new_exp, seed{N}_seed{N}) layouts, across the
# baselines_FC / baselines_FC_pace / baselines_FC_ho / baselines_FC_lili groups.
# LiAM/LiAMx egos get --all-has-last-action so make_vec_envs applies
# ConcatLastActionWrapper (obs 79) — the documented ego-side gotcha.
#
# Usage:
#   METHOD=LiAM EGO_KIND=L2     GPU=4 bash eval_regime_vs_heldout.sh
#   METHOD=LiAM EGO_KIND=L1left GPU=4 bash eval_regime_vs_heldout.sh
#   HELDOUT="4 5 6" overrides the seed list.
# Cross-method ego (pair some OTHER agent with METHOD's held-out partners):
#   EGO_PATH=<ckpt> EGO_ARCH=Generalist TAG=xGen_vs_LiAM METHOD=LiAM ... bash eval_regime_vs_heldout.sh
#   (EGO_ARCH picks the ego's arch-required flags, e.g. LiAM/LiAMx need --all-has-last-action;
#    METHOD then only selects the PARTNER pool.)

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"
cd ${PACE_ROOT}

METHOD=${METHOD:?set METHOD=LiAM|LiAMx|Generalist|PACE|LiLi}
EGO_KIND=${EGO_KIND:?set EGO_KIND=L2|L1left}
GPU=${GPU:-0}
TEST_EPISODES=${TEST_EPISODES:-5}
TEST_PASS=${TEST_PASS:-10}

case "$METHOD" in
  LiAM|LiAMx) HELDOUT=${HELDOUT:-"4 5 6"} ;;
  PACE)       HELDOUT=${HELDOUT:-"4 5 6"} ;;
  LiLi)       HELDOUT=${HELDOUT:-"4 6"} ;;
  Generalist) HELDOUT=${HELDOUT:-"4"} ;;
esac

# ---- resolve partner ckpts (held-out right L1 seeds of METHOD) ----
resolve_partner() { # seed -> path or ""
  local s=$1 g
  for g in baselines_FC baselines_FC_pace baselines_FC_ho baselines_FC_lili; do
    local p="logs/Overcooked/june_new_exp/$g/right/${METHOD}/seed${s}/ppo/30000000.pt"
    [ -f "$p" ] && { echo "$p"; return; }
    p="logs/Overcooked/ppo_june_new_exp/$g/right/${METHOD}/seed${s}_seed${s}/ppo/30000000.pt"
    [ -f "$p" ] && { echo "$p"; return; }
  done
}
PARTNERS=""; N=0
for s in $HELDOUT; do
  p=$(resolve_partner $s)
  [ -z "$p" ] && { echo "[warn] $METHOD held-out seed$s not found — skipped"; continue; }
  PARTNERS="${PARTNERS}${p},"; N=$((N+1))
done
[ $N -eq 0 ] && { echo "[abort] no held-out partners found for $METHOD"; exit 1; }
PARTNERS="${PARTNERS%,}"

# ---- resolve ego ckpt ----
resolve_ego() {
  local g p
  if [ "$EGO_KIND" = "L2" ]; then
    for g in L2_baselines_FC L2_baselines_FC_pace L2_baselines_FC_lili; do
      p="logs/Overcooked/ppo_june_new_exp/$g/left/${METHOD}/seed1_seed1/ppo/60000000.pt"
      [ -f "$p" ] && { echo "$p"; return; }
      p="logs/Overcooked/june_new_exp/$g/left/${METHOD}/seed1/ppo/60000000.pt"
      [ -f "$p" ] && { echo "$p"; return; }
    done
  else # L1left
    for g in baselines_FC baselines_FC_pace baselines_FC_lili; do
      p="logs/Overcooked/june_new_exp/$g/left/${METHOD}/seed1/ppo/30000000.pt"
      [ -f "$p" ] && { echo "$p"; return; }
      p="logs/Overcooked/ppo_june_new_exp/$g/left/${METHOD}/seed1_seed1/ppo/30000000.pt"
      [ -f "$p" ] && { echo "$p"; return; }
    done
  fi
}
EGO=${EGO_PATH:-$(resolve_ego)}
[ -z "$EGO" ] && { echo "[abort] ego ckpt not found ($METHOD $EGO_KIND)"; exit 1; }

# ---- ego extra flags follow the EGO's architecture (default: same method) ----
EGO_ARCH=${EGO_ARCH:-$METHOD}
EXTRA=""
case "$EGO_ARCH" in
  LiAM|LiAMx) EXTRA="--all-has-last-action" ;;
esac

OUT=logs/Overcooked/june_new_exp/eval_regime_vs_heldout
mkdir -p "$OUT"
TAG="${TAG:-${METHOD}_${EGO_KIND}}"
echo "[run] $TAG: ego=$EGO vs $N held-out partners (seeds: $HELDOUT) GPU=$GPU"
CUDA_VISIBLE_DEVICES=$GPU python online_test_.py --env-name Overcooked \
  --env_config './environment/overcooked/config/fc_small_test.yaml' \
  --algo ppo --exp-name "june_new_exp/eval_regime_vs_heldout/${TAG}" --log-dir "$OUT/ld_${TAG}" \
  --player-id 0 --policy-dir "$EGO" --seed 1 \
  --history-size 5 --include-current-episode --merge-encoder-computation \
  --train-pool-size $N --test-episodes $TEST_EPISODES --test-pass $TEST_PASS \
  --use-meta-episode --self-obs-mode $EXTRA \
  --partner-policy-paths "$PARTNERS" > "$OUT/${TAG}.log" 2>&1
rc=$?
echo "[done] $TAG rc=$rc — $OUT/${TAG}.log"
exit $rc
