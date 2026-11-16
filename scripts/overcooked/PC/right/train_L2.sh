#!/bin/bash
# PC: Partially Compatible partners: 3-recipe kitchen, each level-1 partner knows two of the three recipes (types LO_PB, LO_TC, PB_TC).
# right: ego in the RIGHT seat (player 1), partners in the left seat.
# Level-2 training (nested regime): METHOD (default Ours) trained against the level-1 pool of the SAME method. 5 seeds by default.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours SEEDS="1 2 3 4 5" GPUS="0 1 2" bash scripts/overcooked/PC/right/train_L2.sh
DOMAIN=PC SIDE=right exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_train_L2.sh" "$@"
