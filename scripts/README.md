# Scripts

All training and evaluation is launched from here. Every script is a bash wrapper that fixes
the domain and the seat and calls one shared implementation in `common/`, so the same code
runs every (domain, seat, method) combination.

## Partner settings: FC, PC, FI

The three Overcooked folders are named after the kind of level-1 partner pool the ego agent
faces. "Compatible" refers to the recipes the partners can prepare.

| code | name | kitchen | level-1 partner pool |
|---|---|---|---|
| **FC** | Fully Compatible | 3-recipe, horizon 40 | every partner was trained against all three level-0 recipes and can prepare any of them. The paper's *homogeneous* setting. |
| **PC** | Partially Compatible | 3-recipe, horizon 40 | each partner was trained against two of the three level-0 recipes; three partner types `LO_PB`, `LO_TC`, `PB_TC` (Lettuce-Onion, Potato-Broccoli, Tomato-Carrot), which overlap pairwise. |
| **FI** | Fully Incompatible | 6-recipe, horizon 80 | each partner was trained against the two level-0 recipes of one ingredient-disjoint type `R1R2`, `R3R4`, `R5R6`; partners of different types share no recipe. The paper's *heterogeneous* setting. |

`lbf/` is Level-Based Foraging (three item types; every level-1 partner can collect any item,
so it corresponds to FC).

## Layout

```
scripts/
  common/                      shared implementations (read these for the full list of variables)
    lib.sh                     environment, method flags, checkpoint lookup, GPU/concurrency helpers
    overcooked_train_L1.sh     level-1 and baseline training vs level-0
    overcooked_train_L2.sh     level-2 training vs a level-1 pool
    overcooked_eval.sh         evaluation vs level-0 (MODE=L0) or held-out level-1 (MODE=L1)
    lbf_train_L1.sh  lbf_train_L2.sh
  overcooked/{FC,PC,FI}/{left,right,bothsides}/
    train_L1.sh                level-1 partner pool of METHOD (default Ours) vs level-0 (default seeds 1-25)
    train_baselines.sh         comparison egos: PACE, LiAM, LiAMx, LiLi, Generalist vs level-0 (default seeds 1-5)
    train_L2.sh                level 2 of METHOD vs its own level-1 pool (default seeds 1-5)
    eval_method_vs_L0.sh       any trained agent vs the level-0 partners
    eval_method_vs_L1.sh       any trained agent vs held-out level-1 partners
  overcooked/llm/              LLM ego agent evaluation (run_llm_suite.sh, run_llm_heldout.sh)
  lbf/{left,right,bothsides}/  train_L1.sh  train_baselines.sh  train_L2.sh
```

Seat folders: `left` trains the ego as player 0 with partners in the right seat, `right` the
opposite, `bothsides` trains one network that plays both seats (`--both-sides`: the seat is
fixed per environment process, half the processes each way) against the both-sides partner
pool. Only the right seat can deliver in Overcooked.

## Methods

`METHOD` / `METHODS` take `Ours`, `PACE`, `LiAM`, `LiAMx`, `LiLi`, `Generalist`. The
architecture flags are defined once in `common/lib.sh` (`method_flags`):

| method | partner model | auxiliary objective |
|---|---|---|
| Ours | permutation-invariant encoder of (own observation, partner action) over the round, mean-pooled into a deterministic 128-d latent; latent-conditioned actor and critic | none, trained end to end through PPO |
| PACE | same as Ours | peer-identification head (cross-entropy over training-partner identity, coefficient 1.0) and classification reward (0.2, 1M-step warm-up, decayed over 25M steps) |
| LiLi | same encoder as Ours | transition prediction on the previous episode (coefficient 1.0) |
| Generalist | GRU over the round (hidden 128, chunks of 20, shared actor-critic) | none |
| LiAM | GRU, own last action appended to the input, state reset every episode | partner observation and action prediction (coefficient 1.0 each) |
| LiAMx | LiAM with the state carried across the episodes of the round | same as LiAM |

PPO settings are shared: 2x128 ReLU MLPs, 1000 steps per update, 4 epochs, 18 minibatches
(4 in LBF), entropy 3e-2, value loss 0.5, gamma 0.99, GAE 0.95, gradient clipping 15.
Learning rate: 1e-3 for every level-1 and baseline run, except Ours in FC (3e-4); 3e-4 for
Ours at level 2 and 1e-3 for the baselines' level 2. Level 1 trains for 30M environment steps,
level 2 for 60M.

## Variables

Every script reads its settings from environment variables, so a run is

```
SEEDS="1 2 3" GPUS="0 1" bash scripts/overcooked/FC/bothsides/train_L1.sh
```

| variable | meaning |
|---|---|
| `SEEDS` or `N_SEEDS` | seed list, or the first N seeds |
| `METHOD` / `METHODS` | method for level-2 and evaluation scripts / list for level-1 scripts |
| `TYPES` | partner types to train (PC: `LO_PB LO_TC PB_TC`, FI: `R1R2 R3R4 R5R6`; `ALL` trains an FI agent against all six recipes) |
| `GPUS`, `MAXJOBS` | GPUs used round-robin, maximum concurrent runs per script |
| `P` | level-0 partner noise (default 0.25 in Overcooked, 0 in LBF) |
| `NUM_ENV_STEPS`, `LR`, `ENV_CONFIG`, `EXTRA_ARGS` | overrides of the defaults above |
| `L1_BASE_DIR`, `TRAIN_SEEDS`/`EVAL_SEEDS` (FC), `TRAIN_SPEC`/`EVAL_SPEC` (PC, FI, LBF) | level-1 pool for level-2 training; spec format `TYPE:1,2,3;TYPE:1,2,3` |
| `LEVEL`, `TYPES`, `EGO_BASE`, `PARTNER_METHOD`, `PARTNER_SEEDS`/`PARTNER_SPEC`, `TEST_PASS`, `TEST_EPISODES` | evaluation: which agent, which partners (default: the same method's held-out seeds 16-25), protocol (default 10 rounds x 5 episodes) |
| `FEED_HISTORY` | write the level-1 partners' interaction history during level-2 training and evaluation as in their own training (default on; off in PC) |
| `PACE_ROOT`, `CONDA_SH`, `CONDA_ENV`, `WANDB_USER` | machine setup, see the top-level README |
| `SMOKE=1` | 24k-step run to check that a script launches |

## Output layout

`train_.py` names its directories from `--exp-name`; the scripts use
`{DOMAIN}/{SIDE}/{LEVEL}/{METHOD}[/{TYPE}]/seed{N}`, which gives

```
logs/Overcooked/ppo_FC/bothsides/L1/Ours/seed3_seed3/ppo/30000000.pt     checkpoints (every 5M steps, latest.pt or individual_latest.pt)
logs/Overcooked/FC/bothsides/L1/Ours/seed3.log                           training log
logs/Overcooked/FI/left/L2/Ours/seed1.log                                level 2
logs/Overcooked/FC/bothsides/eval_vs_L1/Ours_L2_seed1_seat0.log          evaluation ("Final result" line = success rate)
logs/LBF/ppo_LBF/right/L1/Ours/seed1_seed1/ppo/                          LBF
```

The level-2 and evaluation scripts find the partner pool from this layout (the pool of the
same method on the partner side), so the three stages chain without arguments. Set
`L1_BASE_DIR` (or `PARTNER_BASE`) to use a pool stored elsewhere. Every loader accepts the
final numbered checkpoint, `latest.pt` or `individual_latest.pt`.

## Workflow per domain

Each level-1 pool (per type in PC and FI) is split into three disjoint seed groups:

| seeds | role |
|---|---|
| 1-10 | training partners of level 2 |
| 11-15 | evaluation partners during level-2 training (the periodic evaluation in the training log) |
| 16-25 | held-out partners, never seen during training, used by `eval_method_vs_L1.sh` |

1. `train_L1.sh METHOD=M` on the partner side (`right` for a left-seat ego, `left` for a
   right-seat ego, `bothsides` for a both-sides ego) trains M's level-1 pool: 25 seeds for FC and
   LBF, 25 per type (75) for PC and FI, against level-0 partners with noise `P=0.25`.
   Every method that is run through the nested regime needs its own pool.
2. `train_baselines.sh` on the ego side trains the comparison egos: each baseline against the
   level-0 partners, 5 seeds. These agents are never used as partners.
3. `train_L2.sh METHOD=M` on the ego side trains 5 level-2 seeds of M against pool seeds 1-10,
   evaluating during training against seeds 11-15.
4. `eval_method_vs_L1.sh METHOD=M LEVEL=L2` evaluates the level-2 agents and
   `eval_method_vs_L1.sh METHOD=M LEVEL=L1 EGO_BASE=<baseline run dir>` the original-regime
   agents against M's held-out partners (seeds 16-25), method matched; `eval_method_vs_L0.sh`
   evaluates the same agents against the level-0 partners.

The evaluation scripts run the 10 x 5 protocol on the final checkpoint; the history period of
the ego equals the episodes per round. LBF has no separate evaluation script: its held-out
numbers are the during-training evaluation of `train_L2.sh` (`EVAL_SPEC`) and of
`train_L1.sh`/`train_baselines.sh` with `L1_EVAL_SPEC`.

## LLM agents

`overcooked/llm/run_llm_heldout.sh` runs an LLM ego (`eval_llm.py`, high-level skills, per-round
memory) against the held-out level-1 partners with the 10 x 5 protocol; `run_llm_suite.sh` is
the broader model survey. Both need `LLM_BASE_URL` and `LLM_API_KEY` for an OpenAI-compatible
endpoint and take the partner checkpoints from the paths set at the top of the script.
