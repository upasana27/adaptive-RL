# Environments

Each environment has its own README with the full description (map, recipes or item
types, observation and action spaces, reward, rule-based level-0 partners, partner-pool
loaders, configuration files):

- [`overcooked/`](overcooked/README.md): the two-player Overcooked kitchen in its
  3-recipe (homogeneous partners) and 6-recipe (heterogeneous partners) versions.
- [`lbf/`](lbf/README.md): two-player Level-Based Foraging with typed items.

Both environments expose the same interface to the training code: a maker class
(`OvercookedMaker`, `LBFMaker`) built from a YAML config, a single-agent wrapper that
holds the partner policy (`Overcooked_SingleEnv`, `LBF_SingleEnv`), rule-based
level-0 partners, and loader functions that build the level-0 and level-1 partner
pools from the command-line flags. `policy_common.py` holds the partner wrappers
shared by both.
