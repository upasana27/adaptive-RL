#!/bin/bash
# LBF: Level-Based Foraging, three item types, every level-1 partner can collect any item.
# left: ego in the LEFT seat (player 0), partners in the right seat.
# Level-2 training (nested regime): METHOD (default Ours) against the level-1 pool of the same method. 5 seeds by default.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours SEEDS="1 2 3 4 5" bash scripts/lbf/left/train_L2.sh
SIDE=left exec bash "$(dirname "${BASH_SOURCE[0]}")/../../common/lbf_train_L2.sh" "$@"
