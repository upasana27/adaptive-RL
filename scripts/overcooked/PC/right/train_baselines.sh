#!/bin/bash
# PC: Partially Compatible partners: 3-recipe kitchen, each level-1 partner knows two of the three recipes (types LO_PB, LO_TC, PB_TC).
# right: ego in the RIGHT seat (player 1), partners in the left seat.
# Original-regime training of the BASELINES (PACE with peer-id head, LiAM, LiAMx, LiLi, Generalist) against the level-0 partners (P=0.25).
# These are the comparison egos only ("without regime"); they are never used as partners. A baseline's level-1
# PARTNER pool comes from train_L1.sh METHOD=<baseline>.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHODS="LiAM Generalist" SEEDS="1 2 3 4 5" GPUS="0 1" bash scripts/overcooked/PC/right/train_baselines.sh
DOMAIN=PC SIDE=right METHODS="${METHODS:-PACE LiAM LiAMx LiLi Generalist}" SEEDS="${SEEDS:-1 2 3 4 5}" exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_train_L1.sh" "$@"
