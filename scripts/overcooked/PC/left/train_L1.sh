#!/bin/bash
# PC: Partially Compatible partners: 3-recipe kitchen, each level-1 partner knows two of the three recipes (types LO_PB, LO_TC, PB_TC).
# left: ego in the LEFT seat (player 0), partners in the right seat.
# Level-1 partner pool of METHOD (default Ours; any of Ours, PACE, LiAM, LiAMx, LiLi, Generalist) trained against the
# rule-based level-0 partners (P=0.25). Per pool (per type in PC and FI): seeds 1-10 = level-2 training partners, 11-15 = evaluation
# partners during level-2 training, 16-25 = held-out partners used only by eval_method_vs_L1.sh.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours N_SEEDS=25 GPUS="0 1" bash scripts/overcooked/PC/left/train_L1.sh
DOMAIN=PC SIDE=left METHODS="${METHODS:-${METHOD:-Ours}}" SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25}" exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_train_L1.sh" "$@"
