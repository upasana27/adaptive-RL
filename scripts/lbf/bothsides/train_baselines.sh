#!/bin/bash
# LBF: Level-Based Foraging, three item types, every level-1 partner can collect any item.
# bothsides: one ego network trained for BOTH seats (--both-sides), partners from the both-sides pool.
# Original-regime training of the BASELINES (PACE with peer-id head, LiAM, LiAMx, LiLi, Generalist) against the level-0 partners (P=0.25).
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHODS="LiAM Generalist" SEEDS="1 2 3 4 5" bash scripts/lbf/bothsides/train_baselines.sh
SIDE=bothsides METHODS="${METHODS:-PACE LiAM LiAMx LiLi Generalist}" SEEDS="${SEEDS:-1 2 3 4 5}" exec bash "$(dirname "${BASH_SOURCE[0]}")/../../common/lbf_train_L1.sh" "$@"
