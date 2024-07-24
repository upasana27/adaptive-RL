import random

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch.random

from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game.graphic_pipeline import GraphicPipeline
from gym_cooking import Arrdict, Dotdict, arrdict
from gym_cooking.wrappers import SARDConsistencyChecker
from gym_cooking.environment.cooking_zoo import Ingred2ID

class OvercookedMaker:
    def __init__(self, *, mode, horizon, recipes, obs_spaces, obs_range=None, num_agents=2,
                 interact_reward=0.5, progress_reward=1.0, complete_reward=10.0,
                 step_cost=0.1, display=False, **_):
        if not isinstance(obs_spaces, list):
            obs_spaces = [obs_spaces]
        default_desire = [1] * len(Ingred2ID)
        self._env = cooking_zoo.parallel_env(level=mode, num_agents=num_agents, record=False,
                                        max_steps=horizon, recipes=recipes, desire=default_desire, obs_spaces=obs_spaces, obs_range=obs_range,
                                        interact_reward=interact_reward, progress_reward=progress_reward,
                                        complete_reward=complete_reward, step_cost=step_cost)
        self.name = mode
        self.horizon = horizon
        self.players = self._env.possible_agents
        self.action_spaces = Dotdict(self._env.action_spaces) 
        self.observation_spaces = Dotdict((k,Dotdict(obs=v)) for k,v in self._env.observation_spaces.items())
        self.graphic_pipeline = GraphicPipeline(self._env, display=display)
        self.graphic_pipeline.on_init()

    def seed(self, sd):
        np.random.seed(sd)
        torch.random.manual_seed(sd)
        random.seed(sd)

    def get_action_space(self):
        return gym.spaces.Discrete(6)

    def get_observation_space(self):
        # agent observation size
        if isinstance(self._env.unwrapped.obs_size, int):
            return Dotdict(obs=gym.spaces.Box(-1,1,shape=self._env.unwrapped.obs_size))
        else:
            return Dotdict(obs=gym.spaces.Box(0,10,shape=self._env.unwrapped.obs_size))

    def reset(self):
        obs = self._env.reset()
        data = Arrdict()
        for p,k in zip(self.players,obs):
            data[p] = Arrdict(obs=obs[k],
                             reward=np.float32(0),
                             done=False
                            )
        return data

    def step(self,decision):
        actions = {}
        for a,p in zip(self._env.agents,decision.action):
            actions[a] = decision.action[p]
        obs, reward, done, info = self._env.step(actions) 
        data= Arrdict()
        for k in obs.keys():
            data[k] = Arrdict(obs=obs[k],
                             reward=np.float32(reward[k]), 
                             done=done[k]
                            )

        return data, Dotdict(info) 

    def render(self, mode):
        return self.graphic_pipeline.on_render(mode)
    
    def get_env_world(self):
        return self._env.unwrapped.world
    
    def update_desire(self, desire):
        self._env.unwrapped.update_desire(desire)

    @staticmethod
    def make_env(*args,**kwargs):
        env = OvercookedMaker(*args,**kwargs)
        env = SARDConsistencyChecker(env)
        return env

    @staticmethod
    def make_env_from_config(env_config_path, **override_kwargs):
        import yaml
        f = open(env_config_path)
        env_conf = yaml.load(f,Loader=yaml.FullLoader)
        if len(override_kwargs) > 0:
            print('Overriding environment config with', override_kwargs)
            env_conf.update(override_kwargs)
        env = OvercookedMaker(**env_conf)
        env = SARDConsistencyChecker(env)
        return env


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='simple')
    args = parser.parse_args()

    level = 'full_divider_salad_4'
    horizon = 200
    recipes = [
            "LettuceSalad",
            "TomatoSalad",
            "ChoppedCarrot",
            "ChoppedOnion",
            "TomatoLettuceSalad",
            "TomatoCarrotSalad"
            ]

    env = OvercookedMaker.make_env(obs_spaces='dense', mode=level, horizon=horizon, recipes=recipes)
    action_spaces = env.action_spaces
