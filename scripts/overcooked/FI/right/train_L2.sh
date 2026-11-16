#!/bin/bash
# FI: Fully Incompatible partners: 6-recipe kitchen, each level-1 partner knows the two recipes of one ingredient-disjoint type (R1R2, R3R4, R5R6; heterogeneous setting).
# right: ego in the RIGHT seat (player 1), partners in the left seat.
# Level-2 training (nested regime): METHOD (default Ours) trained against the level-1 pool of the SAME method. 5 seeds by default.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours SEEDS="1 2 3 4 5" GPUS="0 1 2" bash scripts/overcooked/FI/right/train_L2.sh
DOMAIN=FI SIDE=right exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_train_L2.sh" "$@"
