#!/bin/bash
# FI: Fully Incompatible partners: 6-recipe kitchen, each level-1 partner knows the two recipes of one ingredient-disjoint type (R1R2, R3R4, R5R6; heterogeneous setting).
# right: ego in the RIGHT seat (player 1), partners in the left seat.
# Evaluate a trained agent (METHOD, LEVEL L1 or L2, SEEDS) against the rule-based level-0 partners, 10 rounds x 5 episodes.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours LEVEL=L2 SEEDS="1 2 3 4 5" bash scripts/overcooked/FI/right/eval_method_vs_L0.sh
DOMAIN=FI SIDE=right MODE=L0 exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_eval.sh" "$@"
