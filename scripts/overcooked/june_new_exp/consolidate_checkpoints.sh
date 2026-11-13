#!/bin/bash
# Post-training checkpoint consolidation.
#
# train_.py auto-prepends "ppo_" to --exp-name, creating checkpoint dirs like:
#   logs/Overcooked/ppo_june_new_exp/{EXP}/{SIDE}[/{TYPE}]/seed{N}_seed{N}/ppo/*.pt
#
# This script moves them into the canonical june_new_exp location and strips
# the redundant _seed{N} suffix:
#   logs/Overcooked/june_new_exp/{EXP}/{SIDE}[/{TYPE}]/seed{N}/ppo/*.pt
#
# Also renames any seed{N}_seed{N} dirs already present in june_new_exp/ (e.g.
# after a manual mv from a previous session).
#
# Usage:
#   bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh <EXP> <SIDE>
#   bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh L2_PC left
#   bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh L2_Loft_FC left
#   bash scripts/overcooked/june_new_exp/consolidate_checkpoints.sh L1_PC left
#
# Safe to re-run: destinations that already exist are skipped.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

EXP="${1:?Usage: $0 <EXP> <SIDE>}"
SIDE="${2:?Usage: $0 <EXP> <SIDE>}"

LOG_BASE="logs/Overcooked"
PPO_ROOT="${LOG_BASE}/ppo_june_new_exp/${EXP}/${SIDE}"
DST_ROOT="${LOG_BASE}/june_new_exp/${EXP}/${SIDE}"

# Move (or rename-in-place) seed{N}_seed{N} dirs in SRC to DST, stripping suffix.
# When SRC == DST this is an in-place rename.
_move_seeds() {
    local src="$1"
    local dst="$2"
    local moved=0 skipped=0
    shopt -s nullglob
    for path in "${src}"/seed*_seed*/; do
        path="${path%/}"
        local name; name="$(basename "${path}")"
        local clean="${name%%_seed[0-9]*}"   # seed1_seed1 → seed1
        local target="${dst}/${clean}"
        if [[ -e "${target}" ]]; then
            echo "    SKIP (exists): ${clean}"
            skipped=$((skipped + 1))
            continue
        fi
        mkdir -p "${dst}"
        echo "    ${name} → ${clean}"
        mv "${path}" "${target}"
        moved=$((moved + 1))
    done
    shopt -u nullglob
    echo "    moved=${moved} skipped=${skipped}"
}

# Process a parent dir that may contain either direct seed dirs or TYPE subdirs.
_process_dir() {
    local src="$1"
    local dst="$2"

    [[ -d "${src}" ]] || return 0

    # Check for direct seed*_seed* children
    local has_seeds=0
    shopt -s nullglob
    for d in "${src}"/seed*_seed*/; do [[ -d "$d" ]] && has_seeds=1; done
    shopt -u nullglob

    if [[ "${has_seeds}" -eq 1 ]]; then
        _move_seeds "${src}" "${dst}"
    fi

    # Check for TYPE subdirs (e.g. LO_PB, LO_TC, PB_TC) — uppercase names that
    # are not themselves seed dirs.
    shopt -s nullglob
    for type_path in "${src}"/*/; do
        type_path="${type_path%/}"
        [[ -d "${type_path}" ]] || continue
        local tname; tname="$(basename "${type_path}")"
        [[ "${tname}" == seed* ]] && continue   # skip already-processed seed dirs
        local has_type_seeds=0
        for d in "${type_path}"/seed*_seed*/; do [[ -d "$d" ]] && has_type_seeds=1; done
        if [[ "${has_type_seeds}" -eq 1 ]]; then
            echo "  type=${tname}:"
            _move_seeds "${type_path}" "${dst}/${tname}"
        fi
    done
    shopt -u nullglob
}

echo "=== consolidate_checkpoints: ${EXP}/${SIDE} ==="

# Step 1: move from ppo_june_new_exp/ → june_new_exp/
if [[ -d "${PPO_ROOT}" ]]; then
    echo "  src: ${PPO_ROOT}"
    echo "  dst: ${DST_ROOT}"
    _process_dir "${PPO_ROOT}" "${DST_ROOT}"
fi

# Step 2: rename seed{N}_seed{N} dirs already sitting in june_new_exp/
if [[ -d "${DST_ROOT}" ]]; then
    echo "  in-place rename in ${DST_ROOT}"
    _process_dir "${DST_ROOT}" "${DST_ROOT}"
fi

echo "=== done ==="
