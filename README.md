# NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming

Official code for the paper

> **NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming**
> Upasana Biswas, Durgesh Kalwar, Subbarao Kambhampati, Sarath Sreedharan
> NeurIPS 2026. [arXiv:2602.17737](https://arxiv.org/abs/2602.17737)

## Overview

Mutual adaptation is a central challenge in human-AI teaming: humans adjust their strategies
in response to the agent they are working with. Agents trained with fixed partners do not
capture this, since their training partners never adapt to them. NestRL is a nested training
regime in which agents at each level are trained against adaptive agents from the level below.

- **Level-0** partners are fixed, rule-based policies. Each of them follows one convention
  (a recipe in Overcooked, an item type in Level-Based Foraging) and does not model its partner.
- **Level-1** partners are trained against the Level-0 partners. They learn to adapt to the
  partner they are paired with, and serve as the adaptive partners.
- The **Level-2** agent is trained against a pool of Level-1 partners. It has to coordinate
  with partners that are themselves adapting to it.

All agents are evaluated with held-out Level-1 partners that they do not interact with during
training, and with the fixed Level-0 partners. The baselines (PACE, LiAM, LiAMx, LiLi and
Generalist) are trained with the Level-0 partners, as in their original protocols, and can
also be trained with the nested regime.

## Repository

The repository follows the steps above. The environments define the tasks and the Level-0
partners, the scripts train and evaluate each level, and the analysis reads the evaluation
rollouts.

| path | content | documentation |
|---|---|---|
| `environment/` | Overcooked with three and with six recipes, Level-Based Foraging, the rule-based Level-0 partners and the loaders for Level-1 partner pools | [environment/README.md](environment/README.md) |
| `scripts/` | launch scripts for training and evaluation, organized by setting and by the seat of the agent | [scripts/README.md](scripts/README.md) |
| `analysis/` | behavior analysis of evaluation rollouts (who commits to a recipe first, which recipes are prepared, changes over a round) and statistics | [analysis/README.md](analysis/README.md) |
| `llm_agent/`, `eval_llm.py` | evaluation of an LLM as the agent, with the same partners and protocol | [llm_agent/README.md](llm_agent/README.md) |
| `learning/`, `train_.py`, `online_test_.py` | models, PPO and the training and evaluation entry points called by the scripts | |

We consider three partner settings in Overcooked, which differ in the recipes the Level-1
partners are able to prepare, and one setting in Level-Based Foraging:

| setting | environment | Level-1 partners |
|---|---|---|
| `FC` (homogeneous) | Overcooked, three recipes | every partner can prepare all three recipes |
| `PC` | Overcooked, three recipes | each partner can prepare two of the three recipes |
| `FI` (heterogeneous) | Overcooked, six recipes | each partner can prepare the two recipes of one of three types with disjoint ingredients |
| `lbf` | Level-Based Foraging, three item types | every partner can collect any item |

In each setting the agent can be trained for the left seat, the right seat, or both seats
(`left`, `right`, `bothsides`).

## Installation

Requirements: Linux, [conda](https://docs.conda.io), and a CUDA-capable GPU (training on CPU
is possible but slow). From the repository root:

```bash
# 1. environment (the pinned setuptools and wheel are needed to build gym 0.21)
conda create -n pace python=3.8 setuptools=65.6.3
conda activate pace

# 2. PyTorch: pick the build for your CUDA version (https://pytorch.org)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# 3. dependencies
conda install scikit-learn==1.0.2 -c conda-forge
pip install wheel==0.38.4 PettingZoo==1.9.0 pygame==2.0.1 wandb==0.14.0 tqdm==4.65.0 \
    stable-baselines3==1.7.0 seaborn==0.13.0 pyglet==1.5.27 scipy pyyaml

# 4. the Overcooked environment, in editable mode
pip install -e environment/overcooked/gym_cooking/rebar
pip install -e environment/overcooked
```

The same commands are in `install.sh`. To check the installation, run a short training job
(24k environment steps, about two minutes):

```bash
SMOKE=1 SEEDS=1 bash scripts/overcooked/FC/bothsides/train_L1.sh
```

It should end with `all Level-1 runs finished` and leave a checkpoint under
`logs/Overcooked/ppo_smoke_FC/bothsides/L1/Ours/seed1_seed1/ppo/`.

The launch scripts read a few environment variables:

| variable | default | meaning |
|---|---|---|
| `CONDA_ENV` | `pace` | conda environment the scripts activate |
| `CONDA_SH` | `$HOME/anaconda3/etc/profile.d/conda.sh` | location of conda's shell setup |
| `PACE_ROOT` | the repository root | root for `logs/` and checkpoint lookup |
| `WANDB_USER` | unset | Weights & Biases entity; logging is off when unset |
| `GPUS`, `MAXJOBS` | `0`, 3 to 4 | GPUs used round-robin and concurrent runs per script |

## Running the experiments

Every setting and seat has the same five scripts under `scripts/overcooked/<setting>/<seat>/`
(three under `scripts/lbf/<seat>/`). We give the full sequence for the homogeneous setting
with one agent trained for both seats.

```bash
D=scripts/overcooked/FC/bothsides

# 1. Train the pool of Level-1 partners against the Level-0 partners.
bash $D/train_L1.sh

# 2. Verify the pool: every Level-1 partner should coordinate with the Level-0 partners.
LEVEL=L1 SEEDS="$(seq -s ' ' 1 25)" bash $D/eval_method_vs_L0.sh

# 3. Train the baselines against the Level-0 partners.
bash $D/train_baselines.sh

# 4. Train the Level-2 agent (NestRL) against the Level-1 partners.
bash $D/train_L2.sh

# 5. Evaluate with the held-out Level-1 partners and with the Level-0 partners.
LEVEL=L2 bash $D/eval_method_vs_L1.sh
LEVEL=L2 bash $D/eval_method_vs_L0.sh
```

By default, a pool has 25 Level-1 partners (per partner type in `PC` and `FI`): 10 are used
to train the Level-2 agent, 5 to evaluate it during training, and 10 are held out for the
final evaluation. Level-2 agents and baselines are trained with 5 seeds. Level-1 agents are
trained for 30M environment steps and Level-2 agents for 60M. `SEEDS`, `GPUS` and `MAXJOBS`
fit the runs to the available hardware, and `METHOD` runs any of the baselines through the
same steps. All variables are listed in [scripts/README.md](scripts/README.md).

**Evaluation.** An evaluation consists of 10 rounds of 5 episodes with the same partner.
The interaction history of the agent is cleared between rounds, as during training. The
success rate of each run is printed at the end of the evaluation script.

**Behavior analysis.** Adding `SAVE_ROLLOUTS=1` to an evaluation command saves the events of
both agents at every step. The scripts in `analysis/` use these rollouts to measure whether
an agent waits for its partner or leads, which recipes it prepares, and how its behavior
changes over the episodes of a round.

**LLM agents.** `scripts/overcooked/llm/run_llm_heldout.sh` evaluates an LLM as the agent
with the held-out Level-1 partners. It requires an OpenAI-compatible endpoint
(`LLM_BASE_URL`, `LLM_API_KEY`).

## Citation

```bibtex
@inproceedings{biswas2026nestrl,
  title     = {NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming},
  author    = {Biswas, Upasana and Kalwar, Durgesh and Kambhampati, Subbarao and Sreedharan, Sarath},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2602.17737},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

This repository builds on the PACE codebase and on the following projects. We thank their authors.

- PACE: https://github.com/linkct/PACE
- PPO: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- GSCU: https://github.com/YeTianJHU/GSCU
- Overcooked: https://github.com/51616/marl-lipo
- Level-Based Foraging: https://github.com/semitable/lb-foraging
- MPE: https://github.com/openai/multiagent-particle-envs
