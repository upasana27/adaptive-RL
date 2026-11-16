#!/bin/bash
# PC: Partially Compatible partners: 3-recipe kitchen, each level-1 partner knows two of the three recipes (types LO_PB, LO_TC, PB_TC).
# left: ego in the LEFT seat (player 0), partners in the right seat.
# Evaluate a trained agent (METHOD, LEVEL L1 or L2, SEEDS) against the HELD-OUT level-1 partners of the same method (method matched; PARTNER_METHOD overrides), 10 rounds x 5 episodes.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours LEVEL=L2 SEEDS="1 2 3 4 5" PARTNER_SEEDS="16 17 18 19 20 21 22 23 24 25" bash scripts/overcooked/PC/left/eval_method_vs_L1.sh
DOMAIN=PC SIDE=left MODE=L1 exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_eval.sh" "$@"
