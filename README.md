# NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming

This repository contains the code for **NestRL**, a nested training framework that enables reinforcement learning agents to coordinate with both fixed and adaptive partners in cooperative multi-agent environments.

---

## Overview

NestRL introduces a hierarchical training regime in which agents are exposed to partners of increasing adaptivity during training. By nesting training against fixed (Level-0) partners inside an outer loop that trains against adaptive (Level-1) partners, NestRL produces **Level-2 agents** capable of robust coordination with both non-adaptive and adaptive teammates — including real humans.

## Experimental Setup

### Domain

We evaluate NestRL in the **Overcooked** collaborative cooking environment ([Carroll et al., 2019](https://arxiv.org/abs/1910.05789)), a standard benchmark for human–AI coordination. We adopt a multi-recipe, required-coordination variant ([Charakorn et al., 2023](https://arxiv.org/abs/2301.08301)) in which agents must complete *one of three* distinct recipes per episode:

| Recipe | Ingredients |
|--------|------------|
| R1 | Lettuce + Onion Salad |
| R2 | Tomato + Carrot Salad |
| R3 | Potato + Broccoli Salad |

### Partners

- **Fixed (Level-0) Partners:** Rule-based policies specialized to a single recipe. These are non-adaptive and used in nested training and evaluation.
- **Adaptive (Level-1) Partners:** Generated via RL with random seeds, producing held-out adaptive partners not seen during Level-2 training. Strong performance against these partners indicates robustness to adaptive behaviors.

### Baselines

We compare against state-of-the-art adaptive MARL baselines:

| Method | Description |
|--------|-------------|
| **LIAM** ([Papoudakis et al., 2021](https://arxiv.org/abs/2104.09127)) | Learns a partner model via auxiliary prediction of partner observations and actions |
| **LILI** ([Xie et al., 2021](https://arxiv.org/abs/2003.06094)) | Encodes partner behavior implicitly through cross-episode context |
| **Generalist** | Leverages cross-episode adaptation without explicit opponent modeling |
| **PACE** | Performs context-aware exploration with a peer-identification auxiliary objective |

All baselines are trained against fixed (non-adaptive) partners and do not encounter adaptive behaviors during training, isolating the effect of nested exposure to adaptive policies.

---

## Installation

**Requirements:** Python 3.8, PyTorch 1.12.1, CUDA 11.3

```bash
# Create conda environment
conda create -n nestrl python=3.8 setuptools=65.6.3
conda activate nestrl

# Install PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda install torchaudio==0.12.1 -c pytorch

# Install dependencies
conda install scikit-learn==1.0.2 -c conda-forge
pip install pip==23.0.1
pip install wheel==0.38.4 PettingZoo==1.9.0 pygame==2.0.1 wandb==0.14.0 \
    tqdm==4.65.0 stable-baselines3==1.7.0 seaborn==0.13.0 pyglet==1.5.27

# Install environment packages
cd environment/overcooked/gym_cooking/rebar && pip install -e .
cd ../.. && pip install -e .
cd ../..
```

See `install.sh` for the full installation script. You may need to run each line individually.

---

## Training

Training scripts for all methods are provided in `scripts/`. Each script calls `train_.py` with the appropriate hyperparameters.

### NestRL (Ours)

```bash
bash scripts/overcooked/pace.sh
```

### Baselines

```bash
# LIAM
bash scripts/overcooked/liam.sh <seed>

# LILI
bash scripts/overcooked/lili.sh <seed>

# Generalist
bash scripts/overcooked/generalist.sh <seed>
```

Additional environments (Kuhn Poker, Predator-Prey) have corresponding scripts in `scripts/kuhn_poker/` and `scripts/predator_prey/`.

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `--env-name` | Environment: `Overcooked`, `KuhnPoker`, `MPE` |
| `--algo` | RL algorithm (`ppo`) |
| `--num-env-steps` | Total training steps (default: 30M) |
| `--train-pool-size` | Number of training partners in the pool |
| `--latent-training` | Enable latent context encoder |
| `--seed` | Random seed |
| `--wandb-user-name` | Weights & Biases username for logging |

---

## Evaluation

Evaluate trained checkpoints against held-out partners:

```bash
python online_test_.py --policy-dir <path_to_checkpoint> --log-dir <output_dir> ...
```

---

## Code Structure

```
├── train_.py                  # Main training entry point
├── online_test_.py            # Evaluation of trained checkpoints
├── evaluation_.py             # Evaluation utilities
├── vqvae_functions.py         # VQ-VAE utilities
├── learning/
│   ├── arguments.py           # Training hyperparameters and CLI args
│   ├── model.py               # Policy network (LatentPolicy) architecture
│   ├── storage_.py            # Rollout and history storage buffers
│   ├── envs.py                # Vectorized environment wrappers
│   ├── distributions.py       # Action distributions
│   ├── utils.py               # Training utilities
│   └── algo/
│       └── ppo_.py            # PPO implementation
├── environment/
│   ├── policy_common.py       # Shared policy utilities
│   ├── overcooked/            # Overcooked environment + configs
│   ├── mpe/                   # Multi-agent Particle Environment
│   └── kuhn_poker/            # Kuhn Poker environment
├── scripts/
│   ├── overcooked/            # Training scripts for Overcooked
│   ├── predator_prey/         # Training scripts for Predator-Prey
│   └── kuhn_poker/            # Training scripts for Kuhn Poker
├── baselines/
│   └── GSCU/                  # GSCU baseline implementation
├── webapp/                    # Human user study web application
│   ├── app.py                 # Flask + SocketIO server
│   ├── models.py              # Model loading for live inference
│   ├── env_wrapper.py         # Environment wrapper for web interface
│   └── templates/             # HTML templates for the study UI
└── logs/                      # Pretrained checkpoints and training logs
```

---

## User Study

The `webapp/` directory contains the web application used for the human user study. It provides a browser-based interface where human participants play the Overcooked game alongside trained AI agents in real time.

---

## Acknowledgements

This repository builds upon the following projects:

- **PPO:** [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
- **GSCU:** [GSCU](https://github.com/YeTianJHU/GSCU)
- **Overcooked:** [marl-lipo](https://github.com/51616/marl-lipo)
- **MPE:** [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)
