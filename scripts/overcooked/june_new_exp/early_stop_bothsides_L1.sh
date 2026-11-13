#!/bin/bash
PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}"
# Early-stop supervisor for the FC both-sides L1 baselines (user decision 2026-10-22):
# a run whose HELD-OUT eval success has been >= THRESH on CONSEC consecutive evals is good enough;
# kill it (by PID) so the launcher starts the next queued seed, and expose its latest.pt as
# ppo/30000000.pt so the L2 loaders (which expect that filename) can use it. Marker file records the step.
# External on purpose: the running launcher script must not be edited, and queued runs have no --early-stop flag.
GROUP=${GROUP:-baselines_FC_bothsides_p025}; SIDE=${SIDE:-bothsides}
THRESH=${THRESH:-0.93}; CONSEC=${CONSEC:-3}; PERIOD=${PERIOD:-300}
cd ${PACE_ROOT}
LOGDIR=logs/Overcooked/june_new_exp/$GROUP/$SIDE
echo "[$(date '+%F %T')] early-stop supervisor: $GROUP/$SIDE  eval>=${THRESH} x${CONSEC}  every ${PERIOD}s"
while true; do
  for lg in $LOGDIR/*_seed*.log; do
    [ -f "$lg" ] || continue
    name=$(basename $lg .log); m=${name%_seed*}; s=${name##*_seed}
    exp="june_new_exp/$GROUP/$SIDE/$m/seed$s"
    raw=logs/Overcooked/ppo_june_new_exp/$GROUP/$SIDE/$m/seed${s}_seed${s}/ppo
    [ -f "$raw/30000000.pt" ] && continue                      # finished (or already handled)
    # last CONSEC held-out evals
    evs=$(grep -oE "success_rate': [0-9.]+" "$lg" | tail -n $CONSEC | grep -oE '[0-9.]+$')
    n=$(echo "$evs" | grep -c .)
    [ "$n" -lt "$CONSEC" ] && continue
    ok=$(echo "$evs" | awk -v t=$THRESH 'BEGIN{o=1} {if ($1+0 < t) o=0} END{print o}')
    [ "$ok" != "1" ] && continue
    # find the live train_.py for this exp (exact --exp-name match, no self-match)
    pid=""; for f in /proc/[0-9]*/cmdline; do c=$(tr '\0' ' ' < $f 2>/dev/null); case "$c" in python*train_.py*"--exp-name $exp "*) pid=$(basename $(dirname $f)); break;; esac; done 2>/dev/null
    [ -z "$pid" ] && continue
    steps=$(grep -oE 'num timesteps [0-9]+' "$lg" | tail -1 | grep -oE '[0-9]+$')
    # latent methods (PACE/LiLi) write latest.pt; joint-training ones (LiAM/LiAMx/Generalist) write individual_latest.pt
    src=""; for cand in latest.pt individual_latest.pt; do [ -f "$raw/$cand" ] && { src="$raw/$cand"; break; }; done
    [ -n "$src" ] || { echo "[$(date '+%T')] $name qualifies but no rolling checkpoint yet -- skip"; continue; }
    cp "$src" "$raw/30000000.pt" && echo "early-stopped $(date '+%F %T') at step $steps, evals: $(echo $evs | tr '\n' ' '); 30000000.pt is a copy of $(basename $src)" > "$raw/EARLY_STOPPED_at_${steps}.txt"
    kill -TERM $pid; sleep 5; kill -0 $pid 2>/dev/null && kill -KILL $pid
    echo "[$(date '+%F %T')] STOPPED $name at ${steps} steps (last $CONSEC evals: $(echo $evs | tr '\n' ' ')) -> $raw/30000000.pt"
  done
  # stop when the launcher is gone and nothing is left running
  if ! ls /proc/[0-9]*/cmdline 2>/dev/null | xargs -I{} sh -c 'tr "\0" " " < {} 2>/dev/null' 2>/dev/null | grep -q "train_.py.*$GROUP/$SIDE"; then
    if ! ls /proc/[0-9]*/cmdline 2>/dev/null | xargs -I{} sh -c 'tr "\0" " " < {} 2>/dev/null' 2>/dev/null | grep -q "bash.*train_baselines_FC_right.sh"; then echo "[$(date '+%F %T')] nothing left -> exit"; exit 0; fi
  fi
  sleep $PERIOD
done
