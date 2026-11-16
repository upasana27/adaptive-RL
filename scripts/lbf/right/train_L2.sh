#!/bin/bash
# LBF: Level-Based Foraging, three item types, every level-1 partner can collect any item.
# right: ego in the RIGHT seat (player 1), partners in the left seat.
# Level-2 training (nested regime): METHOD (default Ours) against the level-1 pool of the same method. 5 seeds by default.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours SEEDS="1 2 3 4 5" bash scripts/lbf/right/train_L2.sh
SIDE=right exec bash "$(dirname "${BASH_SOURCE[0]}")/../../common/lbf_train_L2.sh" "$@"
