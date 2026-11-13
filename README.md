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

Two partner settings share the same code and differ only in the environment
config and the level-0 pool:

- **Homogeneous (FC)**: 3 recipes, 6 ingredients, 7x8 kitchen, horizon 40,
  config `environment/overcooked/config/fc_small_test.yaml`.
- **Heterogeneous (FI)**: 6 recipes in 3 ingredient-disjoint types, 12
  ingredients, 7x11 kitchen, horizon 80, config
  `environment/overcooked/config/fc_small_6recipe_h80.yaml`
  (level-2 shaping variant: `fc_small_6recipe_h80_b3c3_pair_c80_plate5.yaml`).
- **PC**: level-1 partners specialised to recipe subsets of the homogeneous domain.

Scripts live under `scripts/overcooked/june_new_exp/`:

| stage | homogeneous (FC) | heterogeneous (FI) | PC |
|---|---|---|---|
| level 1, one seat | `train/left/train_L1_FC.sh`, `train/right/train_baselines_FC_right.sh` | `train/left/train_L1_FI_p05.sh`, `supervise_L1_FI.sh`, `train/right/train_baselines_FI.sh`, `train/left/train_baselines_FI_left_ALL.sh` | `train/left/train_L1_PC.sh`, `train/right/train_L1_PC_p05_peerid_batched.sh`, `scripts/overcooked/train_L1_specialized.sh` |
| level 1, both seats | `train/train_L1_bothsides.sh` | `train/train_L1_bothsides.sh` (set `ENV_CONFIG`) | |
| level 2 | `train/train_L2_bothsides_FC.sh`, `train/left/train_L2_baselines_FC.sh` | `train/left/train_L2_FI.sh` | `train/left/train_L2_PC.sh` |
| evaluation | `eval/bothsides/*.sh`, `eval/left/eval_regime_vs_heldout.sh`, `eval/left/eval_baselines_L2_vs_heldout_FC.sh`, `eval/{left,right}/eval_L1_FC_vs_L0.sh`, `eval/eval_L1_bothsides_crossplay.sh` | `eval/right/eval_L1_FI_vs_L0.sh` | `eval/right/eval_L1_PC_vs_L0.sh` |

`consolidate_checkpoints.sh` and `early_stop_bothsides_L1.sh` are helpers used by
the launchers. Each script documents its environment-variable overrides in its header.

Evaluation follows the training protocol: 10 rounds of 5 episodes
(`online_test_.py --test-pass 10 --test-episodes 5`), with the ego's history
cleared between rounds.

## Level-Based Foraging

`environment/lbf/` implements the two-agent typed-fruit LBF variant. Scripts are
under `scripts/lbf/june_new_exp/` (level-1, level-2 and baselines, config
`environment/lbf/config/lbf_fc_3type.yaml`).

## LLM agents

`eval_llm.py` and `llm_agent/` evaluate an LLM as the ego agent in Overcooked
against trained level-1 partners, with high-level skills (`--actions high`) and
per-round memory (`--memory full`). Any OpenAI-compatible chat endpoint works:
set `LLM_BASE_URL` and `LLM_API_KEY`. `scripts/overcooked/june_new_exp/llm/`
has the suite used in the paper (`run_llm_heldout.sh` is the held-out-partner run).

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
