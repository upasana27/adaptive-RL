#!/bin/bash
# Installation. Run the commands one at a time from the repository root (see README.md).

# 1. Conda environment. The pinned setuptools and wheel versions are required to build gym 0.21,
#    which stable-baselines3 1.7.0 depends on.
conda create -n pace python=3.8 setuptools=65.6.3
conda activate pace

# 2. PyTorch (choose the build that matches your CUDA version, see https://pytorch.org).
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# 3. Python dependencies.
conda install scikit-learn==1.0.2 -c conda-forge
pip install wheel==0.38.4 PettingZoo==1.9.0 pygame==2.0.1 wandb==0.14.0 tqdm==4.65.0 \
    stable-baselines3==1.7.0 seaborn==0.13.0 pyglet==1.5.27 scipy pyyaml

# 4. The Overcooked environment and its helper package, in editable mode.
pip install -e environment/overcooked/gym_cooking/rebar
pip install -e environment/overcooked
