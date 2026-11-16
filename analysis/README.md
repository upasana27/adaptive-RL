# Behaviour analysis

Scripts that turn evaluation rollouts into behavioural statistics: who
commits to a recipe first (waiting versus leading), which recipes an agent goes for and
whether its partner follows, and how behaviour changes across the episodes of a round.
They are the same for every Overcooked setting (FC, PC, FI) and seat.

## 1. Collect rollouts

Add `SAVE_ROLLOUTS=1` to either evaluation script. The evaluation is unchanged; next to each
log a `.jsonl` file is written with the events of both agents at every step.

```bash
SAVE_ROLLOUTS=1 LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L1.sh
# -> logs/Overcooked/FC/bothsides/eval_vs_L1/Ours_L2_seed1_seat0.jsonl, ...
```

`collect_rollouts.py` does the logging (it can also be called directly with the arguments of
`online_test_.py`). Each line is one environment step:

| field | content |
|---|---|
| `env`, `partner` | environment instance and index of its partner in the evaluation pool |
| `ep`, `t` | episode counter of that instance, step within the episode |
| `ego` | seat of the ego agent (0 left, 1 right) |
| `ev_ego`, `ev_par` | events of the ego and of the partner: `picked up Lettuce`, `chopped Lettuce`, `put Lettuce on the Plate`, `DELIVERED ...` |
| `hold_ego`, `hold_par`, `plate`, `pos_ego`, `pos_par`, `a_ego`, `a_par`, `rew`, `done` | state after the step |

`events.py` loads these files, maps ingredients to recipes, and groups episodes into rounds.

## 2. Analyses

Run from the repository root; every script takes the setting and one or more rollout files.

| script | question | output |
|---|---|---|
| `commitment.py` | Does the agent wait for its partner or lead? | fraction of episodes in which the ego commits first, median lead in steps, effect size and Wilcoxon test, success when leading versus following |
| `recipe_distribution.py` | Which recipes does it go for, and does the partner follow? | first-pick and delivered-recipe distributions per seat (with entropy), how often the ego picks first, how often the partner then adopts the ego's recipe |
| `round_position.py` | Does it improve within a round? | success by position 1 to 5 in the round; in FI also how often the first pick is in the partner's type |
| `stats.py` | Table statistics | Wilson interval of a rate, Newcombe interval and chi-square / Fisher test for two rates, coefficient of variation across partners |

```bash
python analysis/commitment.py FC logs/Overcooked/FC/bothsides/eval_vs_L1/Ours_L2_*.jsonl --label "Ours L2 vs held-out L1"
python analysis/recipe_distribution.py FC logs/Overcooked/FC/bothsides/eval_vs_L1/Ours_L2_seed1_seat*.jsonl --pool
python analysis/round_position.py FI R1R2:run_R1R2.jsonl R3R4:run_R3R4.jsonl R5R6:run_R5R6.jsonl
python analysis/stats.py compare 444 500 133 400
```

## Definitions

- **Commitment.** An agent's commitment time in an episode is its first pick-up of an
  ingredient of the recipe the pair ends up preparing (the recipe of the last plated
  ingredient). The first pick-up of any ingredient is not used, because agents often touch
  other ingredients before settling. The ego *leads* when it commits before its partner, or
  when the partner never picks up an ingredient of that recipe. Episodes without a plated
  ingredient are left out.
- **Adoption.** In episodes where the ego picks an ingredient before its partner, the
  partner adopts the ego's recipe if the partner's first pick is of the same recipe.
- **Recipe collapse.** A delivered-recipe distribution concentrated on one recipe (entropy
  near 0 bits) together with high lead and adoption rates: the ego imposes a fixed recipe and
  the partners adapt to it.
- **Position in the round.** The evaluation runs rounds of five episodes with the same
  partner and clears the ego's history between rounds, cutting the episode in progress. Rounds
  are recovered from those cut episodes; only positions 1 to 5 are reported. For
  `round_position.py` in FI, pass each file as `TYPE:path` with the type of the partners in
  that file (evaluate one partner type per run with `PARTNER_SPEC`).

Requires `scipy` in addition to the training environment.
