import numpy as np
import gym
from policy import RuleBasedPolicy
from gym_cooking import Arrdict


class Overcooked_MultiEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.players = self.env.players
        self.player_num = len(self.players)
        self.observation_space = self.env.observation_spaces['player_0'].obs
        self.action_space = self.env.action_spaces['player_0']
        # self.observation_space = [self.env.observation_spaces[f'player_{idx}'].obs for idx in range(self.player_num)]
        # self.action_space = [self.env.action_spaces[f'player_{idx}'] for idx in range(self.player_num)]
        self.last_obs = None

    def episode_length(self):
        return self.env.horizon

    def __getattr__(self, key):

        if key in self.__dict__:
            return self.__dict__[key]
        elif key.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{key}'")
        else:
            return getattr(self.env, key)
    
    def set_desire(self, desire):
        self.env.update_desire(desire)

    def reset(self):
        # reset self id and opponent id
        data = self.env.reset()
        self.last_obs = {k: v.obs for k, v in data.items()}
        return [self.last_obs[f'player_{idx}'] for idx in range(self.player_num)]

    def step(self, action):
        decision = Arrdict()
        for id, player_name in enumerate(self.players):
            decision[player_name] = Arrdict(action=action[id])
        # set env step
        data, info = self.env.step(decision)
        info = info['player_0']
        self.last_obs = {k: v.obs for k, v in data.items()}
        obs = [data[f'player_{idx}'].obs for idx in range(self.player_num)]
        reward = data['player_0'].reward
        done = data['player_0'].done
        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode):
        self.env.render(mode)


    
