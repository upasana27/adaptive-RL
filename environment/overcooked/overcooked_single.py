import numpy as np
import gym
import pygame.event
from environment.overcooked.policy import RuleBasedPolicy
from environment.policy_common import DynamicPolicy
from gym_cooking import Arrdict
import time


class Overcooked_SingleEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self._id = None  # The player controlled by the outer policy
        self.players = self.env.players
        self.opponent = None
        self.observation_space = self.env.observation_spaces['player_0'].obs
        self.action_space = self.env.action_spaces['player_0']
        self.last_obs = None

        self.op_inf_time = 0.0
        self.env_time = 0.0
        self.profiling = False

    def set_desire(self, desire):
        self.env.update_desire(desire)

    def episode_length(self):
        return self.env.horizon

    @staticmethod
    def get_reward_scale():
        return 10.0

    def __getattr__(self, key):

        if key in self.__dict__:
            return self.__dict__[key]
        elif key.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{key}'")
        else:
            return getattr(self.env, key)

    def set_opponent(self, policy):
        self.opponent = policy

    def seed(self, seed=None):
        self.env.seed(seed)

    def set_id(self, id):
        self._id = id
        self.opponent.set_id(1 - self._id)

    def reset(self):
        # reset self id and opponent id
        self.opponent.reset()
        data = self.env.reset()
        self.last_obs = {k: v.obs for k, v in data.items()}
        return self.last_obs[f'player_{self._id}']

    def step(self, action):
        # get oppo action

        if self.profiling:
            opp_inference_st = time.time()
        else:
            opp_inference_st = None

        oppo_action = self.get_oppo_action()

        if self.profiling:
            opp_inference_ed = time.time()
            self.op_inf_time += opp_inference_ed - opp_inference_st
        else:
            opp_inference_ed = None

        # print(action,oppo_action)
        decision = Arrdict()
        for k in self.players:
            if str(self._id) in k:
                decision[k] = Arrdict(action=action)
            else:
                decision[k] = Arrdict(action=oppo_action)
        # set env step
        data, info = self.env.step(decision)
        info = info[f'player_{self._id}']
        info['opponent_obs'] = self.last_obs[f'player_{1 - self._id}']
        info['opponent_act'] = oppo_action
        self.last_obs = {k: v.obs for k, v in data.items()}
        data = data[f'player_{self._id}']

        if self.profiling:
            env_step_ed = time.time()
            self.env_time += env_step_ed - opp_inference_ed

        return data.obs, data.reward, data.done, info

    def print_time(self):
        print('Opponent inference time:', self.op_inf_time, 'Env time:', self.env_time)

    def get_oppo_action(self):
        if (isinstance(self.opponent, RuleBasedPolicy)
                or (isinstance(self.opponent, DynamicPolicy)
                    and isinstance(self.opponent.policy.current_policies[0], RuleBasedPolicy))):
            world = self.env.get_env_world()
            oppo_action = self.opponent(world)
        else:
            oppo_action = self.opponent(self.last_obs[f'player_{1 - self._id}'])
        return oppo_action

    def render(self, mode):
        self.env.render(mode)
        pygame.event.get()
