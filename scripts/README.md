# Scripts

All training and evaluation is launched from here. Every script is a bash wrapper that fixes
the domain and the seat and calls one shared implementation in `common/`, so the same code
runs every (domain, seat, method) combination.

## Partner settings: FC, PC, FI

The three Overcooked folders are named after the kind of Level-1 partner pool the ego agent
faces. "Compatible" refers to the recipes the partners can prepare.

| code | name | kitchen | Level-1 partner pool |
|---|---|---|---|
| **FC** | Fully Compatible | 3-recipe, horizon 40 | every partner was trained against all three Level-0 recipes and can prepare any of them. The *homogeneous* setting. |
| **PC** | Partially Compatible | 3-recipe, horizon 40 | each partner was trained against two of the three Level-0 recipes; three partner types `LO_PB`, `LO_TC`, `PB_TC` (Lettuce-Onion, Potato-Broccoli, Tomato-Carrot), which overlap pairwise. |
| **FI** | Fully Incompatible | 6-recipe, horizon 80 | each partner was trained against the two Level-0 recipes of one ingredient-disjoint type `R1R2`, `R3R4`, `R5R6`; partners of different types share no recipe. The *heterogeneous* setting. |

`lbf/` is Level-Based Foraging (three item types; every Level-1 partner can collect any item,
so it corresponds to FC).

## Layout

```
scripts/
  common/                      shared implementations (read these for the full list of variables)
    lib.sh                     environment, method flags, checkpoint lookup, GPU/concurrency helpers
    overcooked_train_L1.sh     Level-1 and baseline training vs Level-0
    overcooked_train_L2.sh     Level-2 training vs a Level-1 pool
    overcooked_eval.sh         evaluation vs Level-0 (MODE=L0) or held-out Level-1 (MODE=L1)
    lbf_train_L1.sh  lbf_train_L2.sh
  overcooked/{FC,PC,FI}/{left,right,bothsides}/
    train_L1.sh                Level-1 partner pool of METHOD (default Ours) vs Level-0 (default seeds 1-25)
    train_baselines.sh         comparison egos: PACE, LiAM, LiAMx, LiLi, Generalist vs Level-0 (default seeds 1-5)
    train_L2.sh                Level-2 of METHOD vs its own Level-1 pool (default seeds 1-5)
    eval_method_vs_L0.sh       any trained agent vs the Level-0 partners
    eval_method_vs_L1.sh       any trained agent vs held-out Level-1 partners
  overcooked/llm/              LLM ego agent evaluation (run_llm_suite.sh, run_llm_heldout.sh)
  lbf/{left,right,bothsides}/  train_L1.sh  train_baselines.sh  train_L2.sh
```

Seat folders: `left` trains the ego as player 0 with partners in the right seat, `right` the
opposite, `bothsides` trains one network that plays both seats (`--both-sides`: the seat is
fixed per environment process, half the processes each way) against the both-sides partner
pool. Only the right seat can deliver in Overcooked.

## Methods

`METHOD` / `METHODS` take `Ours`, `PACE`, `LiAM`, `LiAMx`, `LiLi`, `Generalist`. The flags of
each method are defined once in `common/lib.sh` (`method_flags`); the baseline implementations
and their flags are those of the PACE codebase.

| method | reference | repository |
|---|---|---|
| PACE | Ma et al., *Fast Peer Adaptation with Context-aware Exploration*, ICML 2024 | https://github.com/linkct/PACE |
| LiAM | Papoudakis et al., *Agent Modelling under Partial Observability for Deep Reinforcement Learning*, NeurIPS 2021 | https://github.com/uoe-agents/LIAM |
| LiAMx | LiAM with the recurrent state carried across the episodes of a round, as implemented in PACE | https://github.com/linkct/PACE |
| LiLi | Xie et al., *Learning Latent Representations to Influence Multi-Agent Interaction*, CoRL 2020 | implementation from https://github.com/linkct/PACE |
| Generalist | recurrent policy trained against the whole partner pool, as implemented in PACE | https://github.com/linkct/PACE |

## Variables

Every script reads its settings from environment variables, so a run is

```
SEEDS="1 2 3" GPUS="0 1" bash scripts/overcooked/FC/bothsides/train_L1.sh
```

| variable | meaning |
|---|---|
| `SEEDS` or `N_SEEDS` | seed list, or the first N seeds |
| `METHOD` / `METHODS` | method for Level-2 and evaluation scripts / list for Level-1 scripts |
| `TYPES` | partner types to train (PC: `LO_PB LO_TC PB_TC`, FI: `R1R2 R3R4 R5R6`; `ALL` trains an FI agent against all six recipes) |
| `GPUS`, `MAXJOBS` | GPUs used round-robin, maximum concurrent runs per script |
| `P` | Level-0 partner noise (default 0.25 in Overcooked, 0 in LBF) |
| `NUM_ENV_STEPS`, `LR`, `ENV_CONFIG`, `EXTRA_ARGS` | overrides of the defaults above |
| `L1_BASE_DIR`, `TRAIN_SEEDS`/`EVAL_SEEDS` (FC), `TRAIN_SPEC`/`EVAL_SPEC` (PC, FI, LBF) | Level-1 pool for Level-2 training; spec format `TYPE:1,2,3;TYPE:1,2,3` |
| `LEVEL`, `TYPES`, `EGO_BASE`, `PARTNER_METHOD`, `PARTNER_SEEDS`/`PARTNER_SPEC`, `TEST_PASS`, `TEST_EPISODES` | evaluation: which agent, which partners (default: the same method's held-out partners), protocol (default 10 rounds x 5 episodes) |
| `FEED_HISTORY` | write the Level-1 partners' interaction history during Level-2 training and evaluation as in their own training (default on; off in PC) |
| `PACE_ROOT`, `CONDA_SH`, `CONDA_ENV`, `WANDB_USER` | machine setup, see the top-level README |
| `SMOKE=1` | 24k-step run to check that a script launches |

## Output layout

`train_.py` names its directories from `--exp-name`; the scripts use
`{DOMAIN}/{SIDE}/{LEVEL}/{METHOD}[/{TYPE}]/seed{N}`, which gives

```
logs/Overcooked/ppo_FC/bothsides/L1/Ours/seed3_seed3/ppo/30000000.pt     checkpoints (every 5M steps, latest.pt or individual_latest.pt)
logs/Overcooked/FC/bothsides/L1/Ours/seed3.log                           training log
logs/Overcooked/FI/left/L2/Ours/seed1.log                                Level-2
logs/Overcooked/FC/bothsides/eval_vs_L1/Ours_L2_seed1_seat0.log          evaluation ("Final result" line = success rate)
logs/LBF/ppo_LBF/right/L1/Ours/seed1_seed1/ppo/                          LBF
```

The Level-2 and evaluation scripts find the partner pool from this layout (the pool of the
same method on the partner side), so the three stages chain without arguments. Set
`L1_BASE_DIR` (or `PARTNER_BASE`) to use a pool stored elsewhere. Every loader accepts the
final numbered checkpoint, `latest.pt` or `individual_latest.pt`.

## Run-through

Each Level-1 pool (per type in PC and FI) consists of three disjoint groups of partners,
trained with independent seeds: 10 training partners of Level-2, 5 evaluation partners used
during Level-2 training (the periodic evaluation in the training log), and 10 held-out
partners never seen during training, used by `eval_method_vs_L1.sh`.

The whole regime for our method in the homogeneous setting with a both-sides ego, from
nothing to the numbers, is the following sequence (each step waits for the previous one).
Step 2 verifies the partners before they are used: a Level-1 partner that does not reach a high
success rate with its Level-0 partners, or that learned only one recipe of its type, should be
left out of the pool.

```bash
# 1. Level-1 partner pool: 25 partners against the Level-0 partners (P=0.25), 30M steps each
GPUS="0 1 2 3" MAXJOBS=8 bash scripts/overcooked/FC/bothsides/train_L1.sh

# 2. check the pool: every Level-1 partner against the Level-0 partners it was trained with
LEVEL=L1 SEEDS="$(seq -s ' ' 1 25)" bash scripts/overcooked/FC/bothsides/eval_method_vs_L0.sh

# 3. comparison egos: each baseline against the Level-0 partners, 5 seeds each
GPUS="0 1 2 3" MAXJOBS=8 bash scripts/overcooked/FC/bothsides/train_baselines.sh

# 4. Level-2 of our method against the 10 training partners, 5 seeds, 60M steps each
GPUS="0 1 2 3" MAXJOBS=5 bash scripts/overcooked/FC/bothsides/train_L2.sh

# 5. evaluation against the 10 held-out partners (10 rounds x 5 episodes, both seats)
LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L1.sh
LEVEL=L1 METHOD=Generalist EGO_BASE=logs/Overcooked/ppo_FC/bothsides/L1/Generalist \
    PARTNER_METHOD=Ours bash scripts/overcooked/FC/bothsides/eval_method_vs_L1.sh   # one baseline ego vs the same partners
# 6. evaluation of the Level-2 agents against the Level-0 partners
LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L0.sh
```

The success rate of every run is printed at the end of each evaluation script and is on the
`Final result` line of its log under `logs/Overcooked/FC/bothsides/eval_vs_L1/`.
Adding `SAVE_ROLLOUTS=1` to an evaluation command also saves the per-step events of both
agents for the behaviour analyses in [`analysis/`](../analysis/README.md).

Variants:

- **Seat.** For a left-seat ego the pool is trained with `right/train_L1.sh` and the ego with
  `left/train_L2.sh`; the evaluation scripts in `left/` find the right-side pool. The opposite
  for a right-seat ego.
- **Setting.** Replace `FC` by `PC` or `FI`; `train_L1.sh` then trains the pool per partner
  type (`TYPES`), 25 partners each, and the Level-2 and evaluation scripts use all types.
- **Nested regime of a baseline.** `METHOD=LiAM bash .../train_L1.sh` trains LiAM's own pool,
  `METHOD=LiAM bash .../train_L2.sh` its Level-2, and `METHOD=LiAM LEVEL=L2 bash
  .../eval_method_vs_L1.sh` evaluates it against LiAM's held-out partners (method matched).
- **LBF.** `scripts/lbf/{side}/train_L1.sh`, `train_baselines.sh`, `train_L2.sh` with the same
  variables. There is no separate evaluation script: the held-out numbers are the
  during-training evaluation of `train_L2.sh` (`EVAL_SPEC`) and of `train_L1.sh` /
  `train_baselines.sh` with `L1_EVAL_SPEC`.
- **Smaller runs.** `N_SEEDS`, `SEEDS`, `NUM_ENV_STEPS` and the pool variables shrink any step;
  `SMOKE=1` runs a step for 24k steps to check that it launches.

## LLM agents

`overcooked/llm/run_llm_heldout.sh` runs an LLM ego (`eval_llm.py`, high-level skills, per-round
memory) against the held-out Level-1 partners with the 10 x 5 protocol; `run_llm_suite.sh` is
the broader model survey. Both need `LLM_BASE_URL` and `LLM_API_KEY` for an OpenAI-compatible
endpoint and take the partner checkpoints from the paths set at the top of the script.
