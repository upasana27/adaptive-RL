#!/bin/bash
# PC: Partially Compatible partners: 3-recipe kitchen, each level-1 partner knows two of the three recipes (types LO_PB, LO_TC, PB_TC).
# bothsides: one ego network trained for BOTH seats (--both-sides), partners from the both-sides pool.
# Evaluate a trained agent (METHOD, LEVEL L1 or L2, SEEDS) against the rule-based level-0 partners, 10 rounds x 5 episodes.
# Variables: see scripts/README.md and the header of the shared script. Example:
#   METHOD=Ours LEVEL=L2 SEEDS="1 2 3 4 5" bash scripts/overcooked/PC/bothsides/eval_method_vs_L0.sh
DOMAIN=PC SIDE=bothsides MODE=L0 exec bash "$(dirname "${BASH_SOURCE[0]}")/../../../common/overcooked_eval.sh" "$@"
