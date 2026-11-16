#!/bin/bash
# FC: Fully Compatible partners: 3-recipe kitchen, every level-1 partner knows all three recipes (homogeneous setting).
# bothsides: one ego network trained for BOTH seats (--both-sides), partners from the both-sides pool.
# Level-2 training (nested regime): METHOD (default Ours) trained against the level-1 pool of the SAME method. 5 seeds by default.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours SEEDS="1 2 3 4 5" GPUS="0 1 2" bash scripts/overcooked/FC/bothsides/train_L2.sh
DOMAIN=FC SIDE=bothsides exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_train_L2.sh" "$@"
