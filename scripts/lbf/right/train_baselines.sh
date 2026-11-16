#!/bin/bash
# LBF: Level-Based Foraging, three item types, every level-1 partner can collect any item.
# right: ego in the RIGHT seat (player 1), partners in the left seat.
# Original-regime training of the BASELINES (PACE with peer-id head, LiAM, LiAMx, LiLi, Generalist) against the level-0 partners (P=0.25).
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHODS="LiAM Generalist" SEEDS="1 2 3 4 5" bash scripts/lbf/right/train_baselines.sh
SIDE=right METHODS="${METHODS:-PACE LiAM LiAMx LiLi Generalist}" SEEDS="${SEEDS:-1 2 3 4 5}" exec bash "$(dirname "${BASH_SOURCE[0]}")/../../common/lbf_train_L1.sh" "$@"
