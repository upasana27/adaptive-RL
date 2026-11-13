"""Single-agent gym wrapper around LBFMaker, mirroring Overcooked_SingleEnv."""

import gym
import numpy as np

from environment.lbf.policy import LBFRuleBasedPolicy
from environment.policy_common import DynamicPolicy


class LBF_SingleEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self._id = None
        self.players = self.env.players
        self.opponent = None
        self.observation_space = self.env.observation_spaces['player_0']
        self.action_space = self.env.action_spaces['player_0']
        self.last_obs = None

    def episode_length(self):
        return self.env.horizon

    @staticmethod
    def get_reward_scale():
        return 10.0

    def set_desire(self, desire):
        self.env.update_desire(desire)

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{key}'")
        return getattr(self.env, key)

    def set_opponent(self, policy):
        self.opponent = policy

    def seed(self, seed=None):
        self.env.seed(seed)

    def set_id(self, id):
        self._id = id
        self.opponent.set_id(1 - self._id)

    def reset(self):
        if hasattr(self.opponent, 'reset'):
            self.opponent.reset()
        obs = self.env.reset()
        self.last_obs = dict(obs)
        return self.last_obs[f'player_{self._id}']

    def get_oppo_action(self):
        if isinstance(self.opponent, LBFRuleBasedPolicy) or (
            isinstance(self.opponent, DynamicPolicy)
            and isinstance(self.opponent.policy.current_policies[0], LBFRuleBasedPolicy)
        ):
            return int(self.opponent(self.env.get_env_world()))
        return int(self.opponent(self.last_obs[f'player_{1 - self._id}']))

    def step(self, action):
        oppo_action = self.get_oppo_action()
        actions = {}
        for p in self.players:
            if str(self._id) in p:
                actions[p] = int(action)
            else:
                actions[p] = int(oppo_action)
        obs, reward, done, info = self.env.step(actions)
        info = info[f'player_{self._id}']
        info['opponent_obs'] = self.last_obs[f'player_{1 - self._id}']
        info['opponent_act'] = oppo_action
        self.last_obs = dict(obs)
        return obs[f'player_{self._id}'], reward[f'player_{self._id}'], done[f'player_{self._id}'], info

    def render(self, mode='ascii'):
        return self.env.render(mode)
