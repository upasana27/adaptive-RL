# NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming

Official code for the paper

> **NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming**
> Upasana Biswas, Durgesh Kalwar, Subbarao Kambhampati, Sarath Sreedharan
> NeurIPS 2026. [arXiv:2602.17737](https://arxiv.org/abs/2602.17737)

NestRL trains an agent to coordinate with partners that adapt to it. Agents are trained in
levels: each level is trained against a pool of adaptive agents from the level below, and is
evaluated with held-out adaptive partners it has never interacted with. This repository
contains the training regime, the Overcooked and Level-Based Foraging environments, the
baselines, the evaluation of LLM agents, and the behaviour analysis.

## Contents

| path | content | documentation |
|---|---|---|
| `scripts/` | launch scripts for training and evaluation, one folder per setting and seat | [scripts/README.md](scripts/README.md) |
| `environment/` | Overcooked (3-recipe and 6-recipe kitchens) and Level-Based Foraging | [environment/README.md](environment/README.md) |
| `analysis/` | behaviour analysis of evaluation rollouts and table statistics | [analysis/README.md](analysis/README.md) |
| `llm_agent/`, `eval_llm.py` | LLM ego agent and its evaluation | [llm_agent/README.md](llm_agent/README.md) |
| `learning/` | models, interaction-history buffer and PPO | |
| `train_.py`, `online_test_.py` | training and evaluation entry points called by the scripts | |
| `baselines/` | the GSCU baseline of the PACE codebase | |

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

It should end with `all level-1 runs finished` and leave a checkpoint under
`logs/Overcooked/ppo_smoke_FC/bothsides/L1/Ours/seed1_seed1/ppo/`.

The launch scripts read a few environment variables:

| variable | default | meaning |
|---|---|---|
| `CONDA_ENV` | `pace` | conda environment the scripts activate |
| `CONDA_SH` | `$HOME/anaconda3/etc/profile.d/conda.sh` | location of conda's shell setup |
| `PACE_ROOT` | the repository root | root for `logs/` and checkpoint lookup |
| `WANDB_USER` | unset | Weights & Biases entity; logging is off when unset |
| `GPUS`, `MAXJOBS` | `0`, 3 to 4 | GPUs used round-robin and concurrent runs per script |

## Quick start

The full sequence for the homogeneous Overcooked setting, with one agent trained for both
seats (details and all options in [scripts/README.md](scripts/README.md)):

```bash
bash scripts/overcooked/FC/bothsides/train_L1.sh                    # level-1 partner pool
LEVEL=L1 SEEDS="$(seq -s ' ' 1 25)" bash scripts/overcooked/FC/bothsides/eval_method_vs_L0.sh   # verify the pool
bash scripts/overcooked/FC/bothsides/train_baselines.sh             # baselines
bash scripts/overcooked/FC/bothsides/train_L2.sh                    # level-2 agent (NestRL)
LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L1.sh  # held-out adaptive partners
LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L0.sh  # level-0 partners
```

Level-1 runs take 30M environment steps and level-2 runs 60M; use `SEEDS`, `GPUS` and
`MAXJOBS` to fit the runs to your hardware.

## Training regime

Three levels of partners are trained in sequence, each level only against the
previous one (no mixing):

- **Level 0**: rule-based partners, each specialised in one recipe (or item type).
- **Level 1**: PPO against the level-0 pool.
- **Level 2**: PPO against a pool of level-1 seeds. Held-out level-1 seeds are
  used for evaluation.

Every method can be run through the same regime with its own architecture: a method's
level-2 agent is trained against level-1 agents of the same method (`METHOD` in the scripts).

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
the launch scripts (`run_llm_heldout.sh` is the held-out-partner run).

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
