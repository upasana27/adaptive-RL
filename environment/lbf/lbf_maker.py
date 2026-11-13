"""LBF env factory, mirroring OvercookedMaker's role.

Loads a YAML config and instantiates an LBFCore env wrapped with the
single- or multi-agent interfaces expected by `learning/envs.py` and
`train_.py`.
"""

import gym
import numpy as np
import yaml

from environment.lbf.core import LBFCore, NUM_ACTIONS


class LBFMaker:
    def __init__(
        self,
        *,
        grid_size: int = 8,
        horizon: int = 40,
        num_agents: int = 2,
        num_fruit_types: int = 6,
        num_fruits_per_episode: int = 3,
        present_types=None,
        complete_reward: float = 10.0,
        step_cost: float = 0.1,
        align_reward: float = 0.0,
        align_window: int = 5,
        **_,
    ):
        self.horizon = horizon
        self.num_agents = num_agents
        self.num_fruit_types = num_fruit_types
        self.num_fruits_per_episode = num_fruits_per_episode
        self.players = [f'player_{i}' for i in range(num_agents)]

        # `present_types` may be passed as a list of ints in the YAML; coerce
        # to a tuple. Default (None) means "any type may spawn".
        if present_types is not None:
            present_types = tuple(int(t) for t in present_types)

        self._env = LBFCore(
            grid_size=grid_size,
            num_agents=num_agents,
            num_fruit_types=num_fruit_types,
            num_fruits_per_episode=num_fruits_per_episode,
            present_types=present_types,
            horizon=horizon,
            complete_reward=complete_reward,
            step_cost=step_cost,
            align_reward=align_reward,
            align_window=align_window,
        )

        self._obs_size = self._env.observation_size()
        self._action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self._observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        # Allows the single- and multi-env wrappers to look up per-player spaces.
        self.action_spaces = {p: self._action_space for p in self.players}
        self.observation_spaces = {p: self._observation_space for p in self.players}

    # ----------------------------------------------------------- env interface
    def seed(self, sd):
        self._env.seed(sd)

    def set_present_types(self, present_types) -> None:
        self._env.set_present_types(present_types)

    def get_action_space(self):
        return self._action_space

    def get_observation_space(self):
        return self._observation_space

    def reset(self):
        obs = self._env.reset()
        return {p: obs[i] for i, p in enumerate(self.players)}

    def step(self, actions):
        # `actions` is a dict {player_i: int}
        act_list = [int(actions[p]) for p in self.players]
        obs, reward, done, info = self._env.step(act_list)
        obs_d = {p: obs[i] for i, p in enumerate(self.players)}
        rew_d = {p: float(reward) for p in self.players}
        done_d = {p: bool(done) for p in self.players}
        info_d = {p: dict(info) for p in self.players}
        return obs_d, rew_d, done_d, info_d

    def get_env_world(self):
        """Exposed for rule-based policies that introspect agent / fruit positions."""
        return self._env

    def update_desire(self, desire):
        # No-op for LBF (kept for interface parity with overcooked).
        pass

    def render(self, mode='ascii'):
        return self._env.render_ascii()

    # ----------------------------------------------------------- factory hooks
    @staticmethod
    def make_env(**kwargs):
        return LBFMaker(**kwargs)

    @staticmethod
    def make_env_from_config(env_config_path, **override_kwargs):
        with open(env_config_path) as f:
            env_conf = yaml.load(f, Loader=yaml.FullLoader)
        if override_kwargs:
            print('Overriding LBF env config with', override_kwargs)
            env_conf.update(override_kwargs)
        return LBFMaker(**env_conf)
