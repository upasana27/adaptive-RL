# NestRL: Nested Training for Peer Adaptation

Code for the NeurIPS 2026 submission. The repository extends the PACE codebase
(see Acknowledgements) with the nested training regime, the heterogeneous-partner
Overcooked domain, the Level-Based Foraging domain, and the LLM-agent evaluation.

## Installation

See `install.sh` (run the lines one at a time). The scripts assume a conda
environment named `pace`; set `CONDA_ENV` to use a different name and `CONDA_SH`
if conda is not installed under `$HOME/anaconda3`.

All default checkpoint locations are resolved relative to the repository root.
Set `PACE_ROOT` to override this (for example when checkpoints live elsewhere).
Weights & Biases logging is off unless `WANDB_USER` is set.

## Training regime

Three levels of partners are trained in sequence, each level only against the
previous one (no mixing):

- **Level 0**: rule-based, deterministic partners, one recipe (or fruit type) each.
- **Level 1**: PPO against the level-0 pool.
- **Level 2**: PPO against a pool of level-1 seeds. Held-out level-1 seeds are
  used for evaluation.

Every method (ours, PACE with the peer-identification head, LiAM, LiAMx, LiLi,
Generalist) is run through the same regime with its own architecture: a method's
level-2 agent is trained against level-1 agents of the same method.

## Overcooked

Two partner settings share the same code and differ only in the environment config and the
level-0 pool. The scripts name them after the level-1 partner pool the ego faces:

- **FC, Fully Compatible** (homogeneous): 3 recipes, 6 ingredients, 7x8 kitchen, horizon 40;
  every level-1 partner can prepare all three recipes.
- **PC, Partially Compatible**: same kitchen; each level-1 partner knows two of the three recipes.
- **FI, Fully Incompatible** (heterogeneous): 6 recipes in 3 ingredient-disjoint types, 12
  ingredients, 7x11 kitchen, horizon 80; each level-1 partner knows the two recipes of one type.

The environments are described in [`environment/README.md`](environment/README.md); the
scripts, their variables and the output layout in [`scripts/README.md`](scripts/README.md).
For each setting and each seat (`left`, `right`, `bothsides`) there are five scripts:

```
scripts/overcooked/{FC,PC,FI}/{left,right,bothsides}/
    train_L1.sh            level-1 partner pool of any METHOD vs the level-0 partners
    train_baselines.sh     the baselines vs the level-0 partners (comparison egos)
    train_L2.sh            level 2 of any METHOD vs its own level-1 pool (nested regime)
    eval_method_vs_L0.sh   evaluation vs the level-0 partners
    eval_method_vs_L1.sh   evaluation vs held-out level-1 partners
```

Evaluation follows the training protocol: 10 rounds of 5 episodes with the same partner, the
ego's history cleared between rounds.

## Level-Based Foraging

`environment/lbf/` implements the two-agent typed-item LBF variant (see its README). The
scripts under `scripts/lbf/{left,right,bothsides}/` are `train_L1.sh`, `train_baselines.sh`
and `train_L2.sh`, with the same variables as the Overcooked ones.

## LLM agents

`eval_llm.py` and `llm_agent/` evaluate an LLM as the ego agent in Overcooked
against trained level-1 partners, with high-level skills (`--actions high`) and
per-round memory (`--memory full`). Any OpenAI-compatible chat endpoint works:
set `LLM_BASE_URL` and `LLM_API_KEY`. `scripts/overcooked/llm/` has
the suite used in the paper (`run_llm_heldout.sh` is the held-out-partner run).

## Code structure

- `learning/`: context buffer, models (latent encoder, recurrent policies, auxiliary heads) and PPO.
- `environment/overcooked/`: Overcooked environment, rule-based level-0 policies and partner-pool loaders (`policy.py`).
- `environment/lbf/`: Level-Based Foraging.
- `llm_agent/`, `eval_llm.py`: LLM ego agent and its evaluation harness.
- `train_.py`: training entry point. `online_test_.py`: evaluation of trained checkpoints.
- `baselines/`: the GSCU baseline (from PACE). Other baselines share the main codebase.

## Acknowledgements

This repository is built upon the following projects. We sincerely thank their authors.

- PACE: https://github.com/linkct/PACE
- PPO: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- GSCU: https://github.com/YeTianJHU/GSCU
- Overcooked: https://github.com/51616/marl-lipo
- Level-Based Foraging: https://github.com/semitable/lb-foraging
- MPE: https://github.com/openai/multiagent-particle-envs
