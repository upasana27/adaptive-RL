import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(args, env_id, seed, rank, log_dir, allow_early_resets, render_rank=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dmc2gym.make(domain_name=domain, task_name=task)
            env = ClipAction(env)
        elif env_id == 'KuhnPoker':
            from environment.kuhn_poker.kuhn_poker_oppo_hand import KuhnPoker_SingleEnv
            env = KuhnPoker_SingleEnv()
        elif env_id == 'Overcooked':
            from environment.overcooked.overcooked_maker import OvercookedMaker
            override_kwargs = {}
            if hasattr(args, 'horizon') and args.horizon is not None:
                override_kwargs['horizon'] = args.horizon
            if hasattr(args, 'render') and args.render:
                override_kwargs['display'] = True
            if rank == render_rank:
                print('Rendering', rank)
                override_kwargs['display'] = True
            if hasattr(args, 'desire_id') and args.desire_id is not None and args.desire_id > (args.desire_id & -args.desire_id):
                args.env_config = args.env_config.replace('recipe', 'recipe2')
                print('Overriding env config to {} for desire id {}'.format(args.env_config, args.desire_id))
                # raise NotImplementedError('Configs have changed a lot; check the path here and comment this line if you want to continue.')
            env = OvercookedMaker.make_env_from_config(args.env_config, **override_kwargs)
            print("made from config")
            if args.multi_agent > 1:
                from environment.overcooked.overcooked_multi import Overcooked_MultiEnv
                env = Overcooked_MultiEnv(env)
            else:
                from environment.overcooked.overcooked_single import Overcooked_SingleEnv
                env = Overcooked_SingleEnv(env)
        elif env_id == 'MPE':
            from environment.mpe.env import MPE
            assert args.shaped_reward or args.collide_reward, 'At least one of the rewards must be activated'
            env = MPE(args.scenario, args.num_agents, args.num_good_agents, args.horizon, args.obs_radius, args.init_radius,
                      args.shaped_reward, args.collide_reward, args.collide_reward_once, args.watch_tower)
        else:
            raise NotImplementedError

        # step_callbacks are called once per step, right after the environment steps
        # full_reset_callbacks and reset_callbacks are called twice per interaction or episode:
        # - once right before the interaction or episode starts, with no arguments;
        #   returns initialized callback_state
        # - once right after the interaction or episode finishes, with the final callback_state and info;
        #   modifies info and returns nothing
        callback_names = []
        full_reset_callbacks = []
        reset_callbacks = []
        step_callbacks = []

        if args.visit_reward_coef is not None:
            assert args.player_id == 0, 'Visit reward stats only work for player 0'
            def full_reset_callback(callback_state=None, info=None):
                if callback_state is not None:
                    ep_cnt, itr_cnt, itr_first = callback_state
                    info['interaction_stats']['visit_cnt'] = itr_cnt
                    info['interaction_stats']['visit_first_ep'] = itr_first
                else:
                    return 0, 0, None

            def reset_callback(meta_step, callback_state, obs, info=None):
                ep_cnt, itr_cnt, itr_first = callback_state
                if info is not None:
                    info['episode_stats']['visit_cnt'] = ep_cnt
                else:
                    return (0, itr_cnt, itr_first), obs

            def step_callback(meta_step, ep_step, callback_state, obs, rew, done, info):
                ep_cnt, itr_cnt, itr_first = callback_state
                assert 'keypoint_visited' in info
                if info['keypoint_visited'][0]:
                    if args.visit_reward_type == 'step':
                        rew += args.visit_reward_coef
                    if ep_cnt == 0:
                        # First in episode
                        if args.visit_reward_type == 'episode':
                            rew += args.visit_reward_coef
                        if itr_cnt == 0:
                            # First in interaction
                            if args.visit_reward_type == 'interaction':
                                rew += args.visit_reward_coef
                            itr_first = meta_step
                        itr_cnt += 1
                    ep_cnt += 1
                return (ep_cnt, itr_cnt, itr_first), obs, rew, done, info

            callback_names.append('visit_tracker')
            full_reset_callbacks.append(full_reset_callback)
            reset_callbacks.append(reset_callback)
            step_callbacks.append(step_callback)

        if args.shuffle_agents:
            assert args.env_name == 'MPE'

            def full_reset_callback(callback_state=None, info=None):
                if callback_state is None:
                    # Start of an interaction, initializes a new permutation of agents
                    return torch.randperm(args.num_agents - 1)

            def shuffle_obs(perm, obs):
                num_other_agents = args.num_agents - 1
                ego_and_entity, other_vis, other_pos, other_vel = np.split(
                    obs, [10, 10 + num_other_agents, 10 + num_other_agents * 3]
                )
                other_vis = other_vis[perm]
                other_pos = other_pos.reshape(num_other_agents, 2)[perm].reshape(num_other_agents * 2)
                other_vel = other_vel.reshape(num_other_agents, 2)[perm].reshape(num_other_agents * 2)
                return np.concatenate([ego_and_entity, other_vis, other_pos, other_vel])

            def reset_callback(meta_step, callback_state, obs, info=None):
                if obs is not None:
                    return callback_state, shuffle_obs(callback_state, obs)

            def step_callback(meta_step, ep_step, callback_state, obs, rew, done, info):
                return callback_state, shuffle_obs(callback_state, obs), rew, done, info

            callback_names.append('agent_shuffler')
            full_reset_callbacks.append(full_reset_callback)
            reset_callbacks.append(reset_callback)
            step_callbacks.append(step_callback)

        env = TimeStepTrackingWrapper(env, args.all_has_all_time_steps, args.history_size,
                                      callback_names, full_reset_callbacks, reset_callbacks, step_callbacks)

        if args.all_has_rew_done:
            env = ConcatRewardDoneWrapper(env)

        if args.all_has_last_action:
            env = ConcatLastActionWrapper(env)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

        # if args.use_meta_episode:
        #     print(f'Meta episode activated. '
        #           f'This will wrap the environment with a meta-episode wrapper that resets the environment '
        #           f'after {args.history_size} {"episodes" if args.history_use_episodes else "steps"}. '
        #           f'Episodic statistics are still reported per raw episode, handled by the Monitor wrapper.')
        #     env = MetaEpisodeWrapper(env, args.history_size, args.history_use_episodes)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = EpisodicLifeEnv(env)
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env, width=84, height=84)
                env = ClipRewardEnv(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


class TimeStepTrackingWrapper(gym.Wrapper):
    def __init__(self, env, all_has_all_time_steps, max_num_episodes,
                 callback_names, full_reset_callbacks, reset_callbacks, step_callbacks):
        super().__init__(env)
        self.all_has_all_time_steps = all_has_all_time_steps
        assert len(callback_names) == len(full_reset_callbacks) == len(reset_callbacks) == len(step_callbacks)
        self.callback_names = callback_names
        self.full_reset_callbacks = full_reset_callbacks
        self.reset_callbacks = reset_callbacks
        self.step_callbacks = step_callbacks
        self.callback_states = [None] * len(self.full_reset_callbacks)
        if all_has_all_time_steps:
            self.observation_space = Box(
                low=np.concatenate([env.observation_space.low, np.array([0.0, 0.0], dtype=np.float32)]),
                high=np.concatenate([env.observation_space.high, np.array([1.0, 1.0], dtype=np.float32)]),
            )
        self.max_episode_length = env.episode_length() + 1
        self.max_num_episodes = max_num_episodes
        self.ep_step = self.meta_step = None

    def get_callback_state(self, callback_name):
        return self.callback_states[self.callback_names.index(callback_name)]

    def modify_observation(self, obs):
        # print(self.ep_step, self.meta_step)
        if self.all_has_all_time_steps:
            assert 0 <= self.ep_step < self.max_episode_length and 0 <= self.meta_step < self.max_num_episodes
            obs = np.concatenate([
                obs, np.array([self.ep_step / self.max_episode_length, self.meta_step / self.max_num_episodes])
            ])
        return obs

    def full_reset(self):
        self.meta_step = -1
        # Guarantee that this will be followed by a real reset
        self.ep_step = None
        # Get a new state for each tuple of callbacks
        self.callback_states = [full_reset_callback() for full_reset_callback in self.full_reset_callbacks]
        if hasattr(self.env.opponent, 'full_reset'):
            self.env.opponent.full_reset()

    def reset(self):
        if self.max_num_episodes is not None and self.meta_step == self.max_num_episodes - 1:
            self.full_reset()
        self.meta_step += 1
        self.ep_step = 0
        obs = self.env.reset()
        _callback_states = []
        for callback_state, reset_callback in zip(self.callback_states, self.reset_callbacks):
            callback_state, obs = reset_callback(self.meta_step, callback_state, obs)
            _callback_states.append(callback_state)
        self.callback_states = _callback_states
        # Add time stamps at the end in case some callbacks need to modify the observation
        return self.modify_observation(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.ep_step += 1
        # obs = self.modify_observation(obs)
        _callback_states = []
        for callback_state, step_callback in zip(self.callback_states, self.step_callbacks):
            # Modifies reward here
            callback_state, obs, rew, done, info = step_callback(self.meta_step, self.ep_step, callback_state,
                                                                 obs, rew, done, info)
            _callback_states.append(callback_state)
        self.callback_states = _callback_states
        if done:
            # Interaction is over. Maybe put something in the info
            # Modifies info in-place with the same callbacks
            info['episode_stats'] = {}
            for callback_state, reset_callback in zip(self.callback_states, self.reset_callbacks):
                reset_callback(None, callback_state, None, info)
            if self.meta_step == self.max_num_episodes - 1:
                info['interaction_stats'] = {}
                for callback_state, full_reset_callback in zip(self.callback_states, self.full_reset_callbacks):
                    full_reset_callback(callback_state, info)
        return self.modify_observation(obs), rew, done, info


class ConcatRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        # Concatenate last-step reward and current done flag to the observation
        # I.e. for s_t, the observation is [s_t, r_{t-1}, done_t]
        # For s_0, the observation is [s_0, 0, 0]
        super(ConcatRewardDoneWrapper, self).__init__(env)
        self.reward_scale = env.get_reward_scale()  # Approximated max absolute value of reward
        self.observation_space = Box(
            low=np.concatenate([env.observation_space.low, np.array([-1.0, 0.0], dtype=np.float32)]),
            high=np.concatenate([env.observation_space.high, np.array([1.0, 1.0], dtype=np.float32)]),
        )

    def reset(self):
        obs = self.env.reset()
        return np.concatenate([obs, np.array([0.0, 0.0])])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.concatenate([obs, np.array([reward / self.reward_scale, 1.0 if done else 0.0])]), reward, done, info


class ConcatLastActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.action_dim = env.action_space.n
        self.observation_space = Box(
            low=np.concatenate([env.observation_space.low, np.zeros(self.action_dim, dtype=np.float32)]),
            high=np.concatenate([env.observation_space.high, np.ones(self.action_dim, dtype=np.float32)])
        )

    def reset(self):
        obs = self.env.reset()
        return np.concatenate([obs, np.zeros(self.action_dim)])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_action = np.zeros(self.action_dim)
        last_action[action] = 1.0
        return np.concatenate([obs, last_action]), reward, done, info


# class MetaEpisodeWrapper(gym.Wrapper):
#     def __init__(self, env, meta_episode_size, use_episodes):
#         super(MetaEpisodeWrapper, self).__init__(env)
#         self.meta_episode_size = meta_episode_size
#         self.use_episodes = use_episodes
#         self.current_meta_step = None
#
#     def reset(self):
#         self.current_meta_step = 0
#         return self.env.reset()
#
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         if not self.use_episodes:
#             self.current_meta_step += 1
#         if done:
#             if self.use_episodes:
#                 self.current_meta_step += 1
#             if self.current_meta_step >= self.meta_episode_size:
#                 # Real done of the meta-episode. reset() of the meta environment should be called next.
#                 # The episode should end here, instead of a truncation
#                 if 'bad_transition' in info:
#                     del info['bad_transition']
#                 return obs, reward, True, info
#             # Fake done. Reset the internal environment, but keep the meta-episode going.
#             # The episode should keep going, so remove truncation here
#             if 'bad_transition' in info:
#                 del info['bad_transition']
#             return self.env.reset(), reward, False, info
#         return obs, reward, False, info


class PretrainedOpponentWrapper(gym.Wrapper):
    # Pretrained opponent on GPU can't run in the environment process.
    def __init__(self, env, args):
        super(PretrainedOpponentWrapper, self).__init__(env)
        self.opponents = [None] * args.num_processes

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == 'set_opponent':
            assert len(method_args) == 1 and isinstance(indices, int) and len(method_kwargs) == 0
            self.opponents[indices] = method_args[0]
        else:
            return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        opponent_action = self.opponent.predict(obs, deterministic=True)[0]
        return obs, reward, done, info, opponent_action

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_vec_envs(args, env_name, seed, num_processes, log_dir, device, allow_early_resets, num_frame_stack=None,
                  always_use_dummy=False, render_rank=None):
    envs = [
        make_env(args, env_name, seed, i, log_dir, allow_early_resets, render_rank=render_rank)
        for i in range(num_processes)
    ]

    if len(envs) > 1 and not always_use_dummy:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, norm_reward=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        for info_ in info:
            if 'opponent_obs' in info_:
                info_['opponent_obs'] = torch.from_numpy(np.array(info_['opponent_obs'])).float().to(self.device)
            if 'opponent_act' in info_:
                info_['opponent_act'] = torch.from_numpy(np.array(info_['opponent_act'])).to(self.device)
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
