# NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming

Official code for the paper

> **NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming**
> Upasana Biswas, Durgesh Kalwar, Subbarao Kambhampati, Sarath Sreedharan
> NeurIPS 2026. [arXiv:2602.17737](https://arxiv.org/abs/2602.17737)

Mutual adaptation is a central challenge in human-AI teaming: humans adjust their strategies
in response to the agent they are working with. NestRL is a nested training regime in which
agents at each level are trained against adaptive agents from the level below.

- **Level-0** partners are fixed, rule-based policies.
- **Level-1** partners are trained against the Level-0 partners. They are the adaptive partners.
- The **Level-2** agent is trained against a pool of Level-1 partners and is evaluated with
  held-out Level-1 partners that it has not seen during training.

This repository contains the code for training and evaluating NestRL and the baselines in
Overcooked and Level-Based Foraging, the evaluation of LLM agents, and the behavior analysis.

## Contents

| path | content | documentation |
|---|---|---|
| `scripts/` | launch scripts for training and evaluation, one folder per setting and seat | [scripts/README.md](scripts/README.md) |
| `environment/` | Overcooked (3-recipe and 6-recipe kitchens) and Level-Based Foraging | [environment/README.md](environment/README.md) |
| `analysis/` | behavior analysis of evaluation rollouts and statistics | [analysis/README.md](analysis/README.md) |
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

## Quick start

We give the full sequence for Overcooked with three recipes, where one agent is trained for
both seats. All settings and options are described in [scripts/README.md](scripts/README.md).

```bash
bash scripts/overcooked/FC/bothsides/train_L1.sh                    # Level-1 partner pool
LEVEL=L1 SEEDS="$(seq -s ' ' 1 25)" bash scripts/overcooked/FC/bothsides/eval_method_vs_L0.sh   # verify the pool
bash scripts/overcooked/FC/bothsides/train_baselines.sh             # baselines
bash scripts/overcooked/FC/bothsides/train_L2.sh                    # Level-2 agent (NestRL)
LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L1.sh  # held-out adaptive partners
LEVEL=L2 bash scripts/overcooked/FC/bothsides/eval_method_vs_L0.sh  # Level-0 partners
```

Level-1 agents are trained for 30M environment steps and Level-2 agents for 60M. Use `SEEDS`,
`GPUS` and `MAXJOBS` to fit the runs to your hardware.

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
