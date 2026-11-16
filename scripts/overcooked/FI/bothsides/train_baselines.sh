#!/bin/bash
# FI: Fully Incompatible partners: 6-recipe kitchen, each level-1 partner knows the two recipes of one ingredient-disjoint type (R1R2, R3R4, R5R6; heterogeneous setting).
# bothsides: one ego network trained for BOTH seats (--both-sides), partners from the both-sides pool.
# Original-regime training of the BASELINES (PACE with peer-id head, LiAM, LiAMx, LiLi, Generalist) against the level-0 partners (P=0.25).
# These are the comparison egos only ("without regime"); they are never used as partners. A baseline's level-1
# PARTNER pool comes from train_L1.sh METHOD=<baseline>.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHODS="LiAM Generalist" SEEDS="1 2 3 4 5" GPUS="0 1" bash scripts/overcooked/FI/bothsides/train_baselines.sh
DOMAIN=FI SIDE=bothsides METHODS="${METHODS:-PACE LiAM LiAMx LiLi Generalist}" SEEDS="${SEEDS:-1 2 3 4 5}" exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_train_L1.sh" "$@"
