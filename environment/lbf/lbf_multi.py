"""Multi-agent gym wrapper around LBFMaker, mirroring Overcooked_MultiEnv."""

import gym


class LBF_MultiEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.players = self.env.players
        self.player_num = len(self.players)
        self.observation_space = self.env.observation_spaces['player_0']
        self.action_space = self.env.action_spaces['player_0']
        self.last_obs = None

    def episode_length(self):
        return self.env.horizon

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{key}'")
        return getattr(self.env, key)

    def set_desire(self, desire):
        self.env.update_desire(desire)

    def reset(self):
        obs = self.env.reset()
        self.last_obs = dict(obs)
        return [self.last_obs[f'player_{i}'] for i in range(self.player_num)]

    def step(self, action):
        actions = {p: int(action[i]) for i, p in enumerate(self.players)}
        obs, reward, done, info = self.env.step(actions)
        self.last_obs = dict(obs)
        return (
            [obs[f'player_{i}'] for i in range(self.player_num)],
            reward['player_0'],
            done['player_0'],
            info['player_0'],
        )

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode='ascii'):
        self.env.render(mode)
