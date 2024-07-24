# Fast Peer Adaptation with Context-aware Exploration

### Installation

Please see `install.sh` for instructions. Note that you may need to copy each line into the terminal and execute them separately instead of directly running the script itself.

### Running Experiments

See `scripts/` for the training scripts for each environment.

### Code Structure Overview

- `learning/`: Implementation of the context buffer and training algorithms.
- `environment/`: Environment implementations.
- `scripts/`: Training scripts.
- `baselines/`: The GSCU baseline. Other baselines are implemented on the same codebase as our method.
- `train_.py`: Main training entry point.
- `online_test_.py`: Evaluation code of trained checkpoints.

### Acknowledgements

This repository is built upon the following projects. We sincerely thank their authors for the contributions; for references, please refer to the paper.

- PPO: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- GSCU: https://github.com/YeTianJHU/GSCU
- Overcooked: https://github.com/51616/marl-lipo
- MPE: https://github.com/openai/multiagent-particle-envs
