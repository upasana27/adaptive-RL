#!/bin/bash
# FI: Fully Incompatible partners: 6-recipe kitchen, each level-1 partner knows the two recipes of one ingredient-disjoint type (R1R2, R3R4, R5R6; heterogeneous setting).
# left: ego in the LEFT seat (player 0), partners in the right seat.
# Evaluate a trained agent (METHOD, LEVEL L1 or L2, SEEDS) against the HELD-OUT level-1 partners of the same method (method matched; PARTNER_METHOD overrides), 10 rounds x 5 episodes.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours LEVEL=L2 SEEDS="1 2 3 4 5" PARTNER_SEEDS="16 17 18 19 20 21 22 23 24 25" bash scripts/overcooked/FI/left/eval_method_vs_L1.sh
DOMAIN=FI SIDE=left MODE=L1 exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_eval.sh" "$@"
