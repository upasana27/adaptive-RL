import os.path

# Repository root; override with the PACE_ROOT environment variable. Every default
# checkpoint location below is resolved relative to it.
PACE_ROOT = os.environ.get('PACE_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import random
import re
import numpy as np
import torch
import yaml
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_world.abstract_classes import *
from gym_cooking.environment.cooking_zoo import Ingred2ID
import subprocess
from learning.storage_ import PeriodicHistoryStorage
static_objects = ['CutBoard','DeliverSquare','Divider','Plate']
ingredients = ['Lettuce','Potato','Tomato','Onion','Broccoli','Carrot']
dynamic_objects = ['Plate'] + ingredients
Action = ['Put','Take','Chop']
inf = 1e5


class event:
    def __init__(self, action, dynamic_obj, static_obj=None):
        assert ((action=='Take' or action=="Chop") and static_obj is None) or (action=='Put' and static_obj is not None)
        self.action = action
        self.dynamic_obj = dynamic_obj
        self.static_obj = static_obj
        self.available = False
        self.target_location = None
        self.from_divider = False
        self.done = False

    def __str__(self):
        return f'Event(action={self.action}, dynamic_obj={self.dynamic_obj}, static_obj={self.static_obj})'

class PretrainedPolicy:
    def __init__(self,args, level=None,num_agents=None, record=False, 
                max_steps=None, recipes=None, obs_spaces=["dense"],desire=None, obs_range=15,
                interact_reward=0.5, progress_reward=1.0,
                complete_reward=10.0, step_cost=0.05,model_path=None,eval_history=True,agent_id=0, device='cuda'):
        # Lazy init. The model is loaded only when the first observation is received, in the environment process
        # This guarantees that no tensor needs to be moved across processes
        self.model_path = model_path
        self.agent_id = agent_id  # CRITICAL: Must use correct agent_id for L1 partners in L2 training
        self.num_agents = 2
        self.num_test_policies = 1
        self.num_eval_policies = 1
        self.device = device
        self.eval_history = eval_history
        self.model_path = model_path
        self.update_history = True
        self.args = args
        print(self.args.has_meta_time_step)
        print(self.args.include_current_episode)
        print(self.args.last_episode_only)
        print(self.args.merge_encoder_computation)
        print(self.args.pop_oldest_episode)
        print(self.args.self_obs_mode)
        self.last_obs = None
        #self.use_policy_cls_reward = args.policy_cls_reward_coef is not None and inspect_idx is not None
        #if self.use_policy_cls_reward:
         #   assert args.policy_cls_reward_coef == 0.0
        #self.policy_cls_reward_tracker = PolicyClassificationRewardTracker(args, num_eval_policies, num_eval_policies) \
        #     if use_policy_cls_reward else Nonei
        self.dump_latents = False
        self.opp_policies = 1
        self.policy = self.load(model_path)
        if self.policy.is_recurrent:
             self.rnn_states = torch.zeros(self.num_eval_policies, self.policy.rnn_hidden_dim * (1 if self.policy.share_actor_critic else 2)).to('cuda')
        else:
            self.rnn_states = None   
        self.masks = torch.zeros(self.num_eval_policies, 1).to(self.device)       
        if self.args.latent_training:
            self.history = PeriodicHistoryStorage(
                num_processes=self.num_test_policies,
                num_policies=self.num_test_policies,
                history_storage_size=self.args.history_size,
                clear_period=self.args.history_size,
                refresh_interval=1,
                sample_size=None,
                has_rew_done=self.args.has_rew_done,
                max_samples_per_period=None,
                step_mode=False,
                use_episodes=True,
                has_meta_time_step=self.args.has_meta_time_step,
                include_current_episode=self.args.include_current_episode,
                obs_shape=(73,),
                act_shape=tuple(),
                max_episode_length=40,
                merge_encoder_computation=self.args.merge_encoder_computation,
                last_episode_only=self.args.last_episode_only,
                pop_oldest_episode=self.args.pop_oldest_episode,
            )   
            self.history.to('cuda')
        else:
            self.history = None
    def load(self, model_path):
        # print("model path:", model_path)
        while os.path.exists(model_path):
            # print("this is the model path", model_path)
            device = torch.device('cuda')
            #torch.serialization.add_safe_globals([LatentPolicy])
            policy = torch.load(model_path, map_location=device, weights_only=False)
            return policy


    def set_id(self, aid):
        self.agent_id = aid

    def __call__(self, observation, previous_step=None):

        with torch.no_grad():
            if self.eval_history is not None:
                all_agent_indices = torch.tensor([self.agent_id]).to(self.device)  # Use the agent_id passed during initialization
                indices = self.history.get_all_current_indices()
                history = (self.history, (all_agent_indices,) + indices)
                #history = None
                # print(observation.shape)
                # Convert observation to tensor if it's a numpy array
                if isinstance(observation, np.ndarray):
                    observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
                _, action, action_log_prob, rnn_states = self.policy.act(observation, self.rnn_states, self.masks, all_agent_indices, history=history, deterministic=False)
            self.rnn_states = rnn_states
            #policy_pred = self.policy.aux_pol_cls_head(self.policy.last_latents) if use_policy_cls_reward else None
        if str(action.device) == 'cuda':
            action = action.unsqueeze(-1)
        # print(previous_step)
        # if previous_step is not None:
        #     obs,reward,done,info = previous_step
        #     self.history.add(0, self.last_obs, info['self_act'] if 'self_act' in info else None, reward[0] if self.history.has_rew_done else None)
        #     if done:
        #         if self.args.self_obs_mode:
        #             print("this is info type", torch.from_numpy(info['terminal_observation']).float())
        #             self.history.add(0, torch.from_numpy(info['terminal_observation']).float(), None,0.0 if self.history.has_rew_done else None)
        #         self.history.finish_episode(0)
        # if previous_step is not None:
        #     obs,reward,done,infos = previous_step
        #     # print("we got here")
        #     for i,info in enumerate(infos):
        #         if self.update_history:
        #                 if self.args.self_obs_mode:
        #                     # print("self observations")
        #                     if self.args.self_action_mode:
        #                         # print("i come here")
        #                         self.history.add(i, self.last_obs, action, reward[0] if self.history.has_rew_done else None)
        #                     else:
        #                         # print("i comehere")
        #                         print("id",i)
                                
        #                 elif 'self_obs' in info:
        #                     # print("opponent observations")
        #                     eval_history.add(i, info['self_obs'], info['self_act'],
        #                                     reward[i][0] if eval_history.has_rew_done else None)
        #                 if done:
        #                     if self.args.self_obs_mode:
        #                         print(info['terminal_observation'])
        #                         self.history.add(i, torch.from_numpy(info['terminal_observation']).float(), None,
        #                                         0.0 if self.history.has_rew_done else None)
        #                     self.history.finish_episode(i)
        self.last_obs = observation
        # print("action return", action)
        # Convert action tensor back to int for environment compatibility
        return int(action.item())
    def update_opp_history(self, step_data, info):
        if step_data is not None:
            # print("update history")
            reward = step_data.reward
            done = step_data.done
            # print("adding", reward[0])
            # print(self.history.has_rew_done)
            self.history.add(0, self.last_obs, info['self_act'] if 'self_act' in info else None, reward[0] if self.history.has_rew_done else None)
            # print("this is done", type(done))
            if done:
                if self.args.self_obs_mode:
                    # print(info['termination_info'])
                    # this is where the episode ends
                    self.history.add(0,self.last_obs, None,0.0 if self.history.has_rew_done else None)
                self.history.finish_episode(0)
    def reset(self):
        self.policy = self.load(self.model_path)   
        if self.policy.is_recurrent:
             self.rnn_states = torch.zeros(self.num_eval_policies, self.policy.rnn_hidden_dim * (1 if self.policy.share_actor_critic else 2)).to('cuda')
        else:
            self.rnn_states = None   
        self.masks = torch.zeros(self.num_eval_policies, 1).to('cuda')
        if self.args.latent_training:
            self.history = PeriodicHistoryStorage(
                num_processes=self.num_test_policies,
                num_policies=self.num_test_policies,
                history_storage_size=self.args.history_size,
                clear_period=self.args.history_size,
                refresh_interval=1,
                sample_size=None,
                has_rew_done=self.args.has_rew_done,
                max_samples_per_period=None,
                step_mode=False,
                use_episodes=True,
                has_meta_time_step=self.args.has_meta_time_step,
                include_current_episode=self.args.include_current_episode,
                obs_shape=(73,),
                act_shape=tuple(),
                max_episode_length=40,
                merge_encoder_computation=self.args.merge_encoder_computation,
                last_episode_only=self.args.last_episode_only,
                pop_oldest_episode=self.args.pop_oldest_episode,
            )
            self.history.to('cuda')
        else:
            self.history = None
class PretrainedPolicy_test:
    def __init__(self,latent_training, history_size, has_rew_done, has_meta_time_step,  include_current_episode, merge_encoder_computation, last_episode_only, pop_oldest_episode, self_obs_mode, level=None,          num_agents=None, record=False,
                max_steps=None, recipes=None, obs_spaces=["dense"],desire=None, obs_range=15,
                interact_reward=0.5, progress_reward=1.0,
                complete_reward=10.0, step_cost=0.05,model_path=None,eval_history=True,agent_id=0, device='cuda',
                obs_shape=(73,), max_episode_length=40, feed_history=False):
        self._init_obs_shape = obs_shape
        self._max_episode_length = max_episode_length
        # 2026-10-20: opt-in. When True, Overcooked_SingleEnv.step calls observe_step() so this partner's
        # history is filled exactly as in its own training (storage_.py self-obs-mode: own obs + the OTHER
        # agent's action, terminal obs at episode end), and reset() no longer rebuilds the history every
        # episode (PeriodicHistoryStorage starts a new period by itself every history_size episodes).
        # Default False reproduces the old behaviour: the partner's history is never written.
        self.feed_history = feed_history
        # Set by the loaders when feed_history is on, from the (same-architecture) L2's args:
        #  self_action_mode  -> store this partner's OWN action in its history (LiLi) instead of the other's
        #  use_meta_episode  -> recurrent state carries across the 5-episode round (Generalist/LiAMx); if False
        #                       (LiAM) the GRU state is reset at every episode end, as in its own training
        self.self_action_mode = False
        self.use_meta_episode = True
        self._last_own_action = None
        # Lazy init. The model is loaded only when the first observation is received, in the environment process
        # This guarantees that no tensor needs to be moved across processes
        self.model_path = model_path
        self.agent_id = agent_id  # CRITICAL: Must use correct agent_id
        self.num_agents = 2
        self.num_test_policies = 1
        self.num_eval_policies = 1
        self.device = device
        self.eval_history = eval_history
        self.model_path = model_path
        self.update_history = True
        self.self_obs_mode = self_obs_mode
        self.latent_training = latent_training 
        self.history_size = history_size
        self.has_rew_done = has_rew_done
        self.has_meta_time_step = has_meta_time_step 
        self.include_current_episode = include_current_episode
        self.merge_encoder_computation = merge_encoder_computation 
        self.last_episode_only = last_episode_only 
        self.pop_oldest_episode = pop_oldest_episode
        self.last_obs = None
        # For partner policies trained with --all-has-last-action (e.g. LiAM/LiAMx):
        # the eval env does NOT append the last-action one-hot, so we replicate
        # ConcatLastActionWrapper (learning/envs.py) here — append the partner's own
        # last action one-hot to its obs. Auto-detected on first act() and cached.
        self.action_dim = 6  # Overcooked action space is Discrete(6)
        self._needs_last_action = None
        self._last_action_onehot = torch.zeros(1, self.action_dim).to(self.device)
        #self.use_policy_cls_reward = args.policy_cls_reward_coef is not None and inspect_idx is not None
        #if self.use_policy_cls_reward:
         #   assert args.policy_cls_reward_coef == 0.0
        #self.policy_cls_reward_tracker = PolicyClassificationRewardTracker(args, num_eval_policies, num_eval_policies) \
        #     if use_policy_cls_reward else Nonei
        self.dump_latents = False
        self.opp_policies = 1
        self.policy = self.load(model_path)
        if self.policy.is_recurrent:
             self.rnn_states = torch.zeros(self.num_eval_policies, self.policy.rnn_hidden_dim * (1 if self.policy.share_actor_critic else 2)).to('cuda')
        else:
            self.rnn_states = None   
        self.masks = torch.zeros(self.num_eval_policies, 1).to(self.device)       
        # Initialize history if latent_training OR if eval_history is enabled (for L1 partners)
        if self.latent_training or self.eval_history:
            self.history = PeriodicHistoryStorage(
                num_processes=self.num_test_policies,
                num_policies=self.num_test_policies,
                history_storage_size=self.history_size,
                clear_period=self.history_size,
                refresh_interval=1,
                sample_size=None,
                has_rew_done=self.has_rew_done,
                max_samples_per_period=None,
                step_mode=False,
                use_episodes=True,
                has_meta_time_step=self.has_meta_time_step,
                include_current_episode=self.include_current_episode,
                obs_shape=self._init_obs_shape,
                act_shape=tuple(),
                max_episode_length=self._max_episode_length,
                merge_encoder_computation=self.merge_encoder_computation,
                last_episode_only=self.last_episode_only,
                pop_oldest_episode=self.pop_oldest_episode,
            )
            self.history.to('cuda')
        else:
            self.history = None
    def load(self, model_path):
        # print("model path:", model_path)
        while os.path.exists(model_path):
            # print("this is the model path", model_path)
            device = torch.device('cuda')
            #torch.serialization.add_safe_globals([LatentPolicy])
            policy = torch.load(model_path, map_location=device, weights_only=False)
            return policy


    def set_id(self, aid):
        self.agent_id = aid
        # Update tensor when agent_id changes
        # self.agent_indices_tensor = torch.LongTensor([self.agent_id]).to(self.device)  # DISABLED: causes crash with agent_id=1

    def __call__(self, observation, previous_step=None):
        
        # Update history with previous step data BEFORE getting new action
        if previous_step is not None and self.history is not None:
            obs, reward, done, info = previous_step
            # Add previous observation and action to history
            if self.last_obs is not None:
                self.history.add(0, self.last_obs, 
                                info['self_act'] if 'self_act' in info else None, 
                                reward[0] if self.history.has_rew_done else None)
            
            # If episode ended, add terminal observation and finish episode
            if done:
                if self.self_obs_mode and 'terminal_observation' in info:
                    terminal_obs = torch.from_numpy(info['terminal_observation']).float() if isinstance(info['terminal_observation'], np.ndarray) else info['terminal_observation']
                    self.history.add(0, terminal_obs, None, 0.0 if self.history.has_rew_done else None)
                self.history.finish_episode(0)
                # New episode starts next call -> reset the tracked last action to zeros.
                self._last_action_onehot = torch.zeros(1, self.action_dim, device=self.device)

        with torch.no_grad():
            # Convert observation to tensor if it's a numpy array
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            all_agent_indices = torch.arange(1).to(self.device)
            if self.eval_history is not None and self.history is not None:
                indices = self.history.get_all_current_indices()
                history = (self.history, (all_agent_indices,) + indices)
            else:
                history = None

            def _act(obs_in):
                return self.policy.act(obs_in, self.rnn_states, self.masks, all_agent_indices,
                                       history=history, deterministic=False)

            # NOTE: `observation` (self.last_obs / history) stays the original obs; only the
            # policy input is augmented for --all-has-last-action partners.
            if self._needs_last_action:
                obs_in = torch.cat([observation, self._last_action_onehot], dim=-1)
                _, action, action_log_prob, rnn_states = _act(obs_in)
            elif self._needs_last_action is None:
                # First call: probe whether this partner expects the last-action-augmented obs.
                try:
                    _, action, action_log_prob, rnn_states = _act(observation)
                    self._needs_last_action = False
                except RuntimeError as e:
                    if 'shapes cannot be multiplied' not in str(e) and 'size mismatch' not in str(e):
                        raise
                    self._needs_last_action = True
                    obs_in = torch.cat([observation, self._last_action_onehot], dim=-1)
                    _, action, action_log_prob, rnn_states = _act(obs_in)
            else:
                _, action, action_log_prob, rnn_states = _act(observation)
            self.rnn_states = rnn_states
            # Record this action as the partner's "last action" for the next step's augmentation.
            self._last_action_onehot = torch.zeros(1, self.action_dim, device=self.device)
            self._last_action_onehot[0, int(action.reshape(-1)[0].item())] = 1.0
            self._last_own_action = int(action.reshape(-1)[0].item())
            #policy_pred = self.policy.aux_pol_cls_head(self.policy.last_latents) if use_policy_cls_reward else None
        if str(action.device) == 'cuda':
            action = action.unsqueeze(-1)
        # print(previous_step)
        # if previous_step is not None:
        #     obs,reward,done,info = previous_step
        #     self.history.add(0, self.last_obs, info['self_act'] if 'self_act' in info else None, reward[0] if self.history.has_rew_done else None)
        #     if done:
        #         if self.args.self_obs_mode:
        #             print("this is info type", torch.from_numpy(info['terminal_observation']).float())
        #             self.history.add(0, torch.from_numpy(info['terminal_observation']).float(), None,0.0 if self.history.has_rew_done else None)
        #         self.history.finish_episode(0)
        # if previous_step is not None:
        #     obs,reward,done,infos = previous_step
        #     # print("we got here")
        #     for i,info in enumerate(infos):
        #         if self.update_history:
        #                 if self.args.self_obs_mode:
        #                     # print("self observations")
        #                     if self.args.self_action_mode:
        #                         # print("i come here")
        #                         self.history.add(i, self.last_obs, action, reward[0] if self.history.has_rew_done else None)
        #                     else:
        #                         # print("i comehere")
        #                         print("id",i)
                                
        #                 elif 'self_obs' in info:
        #                     # print("opponent observations")
        #                     eval_history.add(i, info['self_obs'], info['self_act'],
        #                                     reward[i][0] if eval_history.has_rew_done else None)
        #                 if done:
        #                     if self.args.self_obs_mode:
        #                         print(info['terminal_observation'])
        #                         self.history.add(i, torch.from_numpy(info['terminal_observation']).float(), None,
        #                                         0.0 if self.history.has_rew_done else None)
        #                     self.history.finish_episode(i)
        self.last_obs = observation
        # print("action return", action)
        # Convert action tensor back to int for environment compatibility
        return int(action.item())
    def update_opp_history(self, step_data, info):
        if step_data is not None:
            reward = step_data.reward
            done = step_data.done
            self.history.add(0, self.last_obs, info['self_act'] if 'self_act' in info else None, reward[0] if self.history.has_rew_done else None)
            # print("this is done", type(done))
            if done:
                if self.self_obs_mode:
                    print(info['termination_info'])
                    self.history.add(0,self.last_obs, None,0.0 if self.history.has_rew_done else None)
                self.history.finish_episode(0)
    def observe_step(self, other_action, done, terminal_obs=None):
        # Only called when feed_history is True. Mirrors storage_.py insert() in self-obs-mode
        # (non self-action-mode): add (obs this partner acted on, other agent's action).
        # Recurrent partners (Generalist etc., trained with --use-meta-episode): keep the GRU state
        # (masks=1) except at each meta-episode start, as train_.py/storage_.py do. Without the feed,
        # masks stay 0 and hxs*masks wipes the hidden state on every step.
        if self.policy.is_recurrent:
            self._feed_episodes = getattr(self, '_feed_episodes', 0) + (1 if done else 0)
            reset_state = done and ((not self.use_meta_episode) or self._feed_episodes % self.history_size == 0)
            self.masks = torch.full((self.num_eval_policies, 1), 0.0 if reset_state else 1.0, device=self.device)
        if self.history is None or self.last_obs is None:
            return
        obs = self.last_obs.reshape(-1)
        act = self._last_own_action if self.self_action_mode else int(np.asarray(other_action).reshape(-1)[0])
        self.history.add(0, obs, act, None)
        if done:
            if self.self_obs_mode and terminal_obs is not None:
                self.history.add(0, torch.as_tensor(np.asarray(terminal_obs), dtype=torch.float32).to(obs.device),
                                 None, None)
            self.history.finish_episode(0)
            self.last_obs = None
    def full_reset(self):
        # Called by TimeStepTrackingWrapper.full_reset() at every meta-episode / eval-pass start, i.e.
        # exactly when the ego's own history is cleared. Feed mode only: start this partner fresh too
        # (otherwise a pass that cuts an episode short leaves a partial episode in its buffer).
        if not self.feed_history:
            return
        self.feed_history = False
        self.reset()
        self.feed_history = True
        self.last_obs = None
        self._feed_episodes = 0
    def reset(self):
        if self.feed_history and self.history is not None:
            # Keep the history across the 5-episode meta-episode (see __init__).
            return
        self.policy = self.load(self.model_path)
        if self.policy.is_recurrent:
             self.rnn_states = torch.zeros(self.num_eval_policies, self.policy.rnn_hidden_dim * (1 if self.policy.share_actor_critic else 2)).to('cuda')
        else:
            self.rnn_states = None
        self.masks = torch.zeros(self.num_eval_policies, 1).to('cuda')
        if self.latent_training:
            self.history = PeriodicHistoryStorage(
                num_processes=self.num_test_policies,
                num_policies=self.num_test_policies,
                history_storage_size=self.history_size,
                clear_period=self.history_size,
                refresh_interval=1,
                sample_size=None,
                has_rew_done=self.has_rew_done,
                max_samples_per_period=None,
                step_mode=False,
                use_episodes=True,
                has_meta_time_step=self.has_meta_time_step,
                include_current_episode=self.include_current_episode,
                obs_shape=self._init_obs_shape,
                act_shape=tuple(),
                max_episode_length=self._max_episode_length,  # was hard-coded 40 (broke H=80 FI partners); default still 40
                merge_encoder_computation=self.merge_encoder_computation,
                last_episode_only=self.last_episode_only,
                pop_oldest_episode=self.pop_oldest_episode,
            )
            self.history.to('cuda')
        else:
            self.history = None

        
class RuleBasedPolicy:
    def __init__(self, policy_type, nav_p, tar_p, rand_p, convention, env_name, support_set=None,
                 ingredient_support_set=None, event_probs=None, ingred_2_id=None,
                 plate_take_requires_deliver=False):
        self._ingred_2_id = ingred_2_id if ingred_2_id is not None else Ingred2ID
        # plate_take_requires_deliver (opt-in, 6recipe_x4 only): only take a completed plate when a
        # DeliverSquare is reachable from this agent's side. Without it, a LEFT-seat L0 grabs the full plate
        # it cannot deliver, puts it back, and the RIGHT L0's plan is invalidated every step (deterministic
        # ping-pong deadlock seen in 8/12 X4 L0xL0 episodes). Default False keeps FC/FI/PC L0s unchanged.
        self.plate_take_requires_deliver = plate_take_requires_deliver
        self.agent_id = None
        assert policy_type in ['minimum', 'medium', 'full', 'specified']
        assert convention in [0,1,2,None]  # only take plate if it is in certain position
        # minimum: only take ingredients to divider, take plate under satisfied recipe to delivery
        # medium: minimum + put ingredients into plate
        # full: all possible events
        self.policy_type = policy_type
        self.nav_p = nav_p  # the probability of moving right/left instead of up/down during navigation
        self.tar_p = tar_p  # the probability of choosing random target location instead of nearest 
        self.rand_p = rand_p # the probability of doing random actions instead of carrying out current plan
        assert (support_set is None and event_probs is None and policy_type != 'specified') or (len(support_set)==len(event_probs) and policy_type == 'specified')
        self.env_name = env_name
        if "divider" in env_name:
            if "large" in env_name:
                print("AT RIGHT SPOT")
                self.divider_loc = [(3,i) for i in range(1,12)]
                interval = 4
                upper = 12
            else:
                self.divider_loc = [(3,i) for i in range(1,6)]
                interval = 2
                upper = 6
            self.convention = [(3,i) for i in range(interval*convention+1,min(interval*(convention+1)+1,upper))] if convention is not None else None
        else:
            raise NotImplementedError
        if support_set is None:
            self.support_set = []
            for action in Action:
                for dynamic_obj in (dynamic_objects if ingredient_support_set is None else (ingredient_support_set + ['Plate'])):
                    if action=="Put":                            
                        for static_obj in static_objects:
                            if dynamic_obj == "Plate" and (static_obj == "Divider" or static_obj == "DeliverSquare"):
                                if self.policy_type == 'full':
                                    self.support_set.append(event(action, dynamic_obj, static_obj))
                                else:
                                    if static_obj == 'DeliverSquare':
                                        # delete task: put plate to divider
                                        self.support_set.append(event(action, dynamic_obj, static_obj))
                            elif dynamic_obj != "Plate" and static_obj != "DeliverSquare":
                                if self.policy_type == 'full':
                                    self.support_set.append(event(action, dynamic_obj, static_obj))
                                elif self.policy_type == 'medium':
                                    # delete task: put ingredient to cutboard
                                    if static_obj != 'CutBoard':
                                        self.support_set.append(event(action, dynamic_obj, static_obj))
                                elif self.policy_type == 'minimum':
                                    # delete task: put ingredient to cutboard/plate
                                    if static_obj == 'Divider':
                                        self.support_set.append(event(action, dynamic_obj, static_obj))
                    elif action=="Chop":
                        if dynamic_obj != "Plate" and self.policy_type == 'full':
                            self.support_set.append(event(action, dynamic_obj))
                    else:
                        self.support_set.append(event(action, dynamic_obj))
            self.event_probs = np.random.rand(len(self.support_set)) + 1
            self.event_probs = self.event_probs/np.sum(self.event_probs)
        else:
            self.support_set = support_set
            self.event_probs = event_probs
        self.ingredient_support_set = ingredients if ingredient_support_set is None else ingredient_support_set
        ingred_id_set = [self._ingred_2_id[ingred] for ingred in self.ingredient_support_set]
        self.ingredient_support_set_id = tuple(int(i in ingred_id_set) for i in range(len(self._ingred_2_id)))
        self.cur_event = None # current event
        #assert (event_probs is None) ^ (event_prio is None)
        # if event_probs is not None:
        #     assert len(support_set) == len(event_probs)
        # if event_prio is not None:
        #     assert len(support_set) == len(event_prio)

    def set_id(self, aid):
        self.agent_id = aid

    def distance(self, loc1, loc2):
        return np.abs(loc1[0]-loc2[0]) + np.abs(loc1[1]-loc2[1])

    def refresh_event(self):
        self.cur_event = None
        for e in self.support_set:
            e.available = False
            e.target_location = None
            e.done = False

    def in_ingredient_support(self, dynamic_obj):
        res = False
        for ingredient in self.ingredient_support_set:
            if isinstance(dynamic_obj, StringToClass[ingredient]):
                res = True
                break
        return res

    def is_reachable(self, agent_loc, target_loc):
        # whether the target_location is reachable from current agent location
        if "divider" in self.env_name:
            divider_x = self.divider_loc[0][0]
            if (agent_loc[0]-divider_x)*(target_loc[0]-divider_x)<0:
                return False
            else:
                return True
        else:
            raise NotImplementedError
    
    def is_movable(self, world, target_loc):
        # whether the target_location is only floor that agent can move onto
        static_obj = world.get_objects_at(target_loc, StaticObject)
        if len(static_obj)>1 or (len(static_obj)==1 and not isinstance(static_obj[0], Floor)):
            return False
        else:
            return True

    def is_event_available(self, e: event, world):
        # return the availability and if available, set the position of the target to the event
        action = e.action
        dynamic_obj = e.dynamic_obj
        static_obj = e.static_obj
        self.loc = world.agents[self.agent_id].location
        if action == "Put":
            if not world.agents[self.agent_id].holding:
                e.available = False
                return False
            elif not isinstance(world.agents[self.agent_id].holding, StringToClass[dynamic_obj]):
                e.available = False
                return False
            elif static_obj == "Divider":
                # if e.target_location is not None:
                #     #print("available check",e.target_location)
                #     if len(world.get_objects_at(e.target_location, DynamicObject))==0 and len(world.get_objects_at(e.target_location, CutBoard))==0:
                #         return True
                min_dist = inf
                avail_loc = []
                for loc in self.divider_loc:
                    if len(world.get_objects_at(loc, DynamicObject))==0 and len(world.get_objects_at(loc, CutBoard))==0:
                        # Can not put plate/ingredient onto a divider where there is ingredient/plate/cutboard on it
                        avail_loc.append(loc)
                        if self.distance(loc, self.loc)<min_dist:
                            min_dist = self.distance(self.loc, loc)
                            e.target_location = loc
                if min_dist == inf:
                    e.available = False
                    return False
                else:
                    e.available = True
                    # if there are multiple available locations, with probability tar_p randomly choose one
                    if np.random.rand()<self.tar_p and len(avail_loc)>1:
                        e.target_location = avail_loc[np.random.randint(len(avail_loc))]
                    return True
            elif static_obj == "Plate":
                if world.agents[self.agent_id].holding.done():
                    # can only put a chopped food into a plate
                    min_dist = inf
                    for plate in world.world_objects['Plate']:
                            loc = plate.location
                            if self.is_reachable(self.loc, loc) and self.distance(self.loc, loc)<min_dist:
                                min_dist = self.distance(self.loc, loc)
                                e.target_location = loc
                    if min_dist == inf:
                        e.available = False
                        return False
                    else:
                        e.available = True
                        return True
                else:
                    return False
            elif static_obj == "CutBoard":
                if world.agents[self.agent_id].holding.done():
                    # cannot chop a chopped food
                    e.available = False
                    return False
                else:
                    min_dist = inf
                    for cutboard in world.world_objects['CutBoard']:
                        loc = cutboard.location
                        if self.is_reachable(self.loc, loc) and len(world.get_objects_at(loc, DynamicObject))==0 and self.distance(self.loc, loc)<min_dist:
                            min_dist = self.distance(self.loc, loc)
                            e.target_location = loc
                    if min_dist == inf:
                        e.available = False
                        return False
                    else:
                        e.available = True
                        return True
            elif static_obj == "DeliverSquare":
                if len(world.agents[self.agent_id].holding.content)==0:
                    # empty plate cannot be put onto deliversquare
                    e.available = False
                    return False
                else:
                    min_dist = inf
                    for deliver_square in world.world_objects['DeliverSquare']:
                        loc = deliver_square.location
                        if self.is_reachable(self.loc, loc) and len(world.get_objects_at(loc, DynamicObject))==0 and self.distance(self.loc, loc)<min_dist:
                            min_dist = self.distance(self.loc, loc)
                            e.target_location = loc
                    if min_dist == inf:
                        e.available = False
                        return False
                    else:
                        e.available = True
                        return True
        elif action == "Chop":
            if world.agents[self.agent_id].holding:
                e.available = False
                return False
            else:
                min_dist = inf
                for food in world.world_objects[dynamic_obj]:
                    loc = food.location
                    if not food.done() and self.is_reachable(self.loc, loc):
                        for cutboard in world.world_objects["CutBoard"]:
                            if loc == cutboard.location:
                                min_dist = self.distance(self.loc, loc)
                                e.target_location = loc
                if min_dist == inf:
                    e.available = False
                    return False
                else:
                    e.available = True
                    return True
        else:
            if world.agents[self.agent_id].holding:
                # Can't take anything when holding something
                e.available = False
                return False
            else:
                min_dist = inf
                for obj in world.world_objects[dynamic_obj]:
                    loc = obj.location
                    if dynamic_obj !="Plate" and len(world.get_objects_at(loc, DynamicObject))>1:
                        # Can not take ingredients already in a plate, instead take the plate
                        continue
                    elif dynamic_obj =="Plate":
                        # Can't take a plate unless to serve a dish
                        in_plate = world.get_objects_at(loc, Food)
                        # recipe not satisfied: for 2-ingredient recipes this is the original "exactly one item" test;
                        # for larger support sets (6recipe_x4, 4 ingredients) the plate must hold them all.
                        _need = len(self.ingredient_support_set)
                        if len(in_plate) == 0 or (len(in_plate) == 1 and _need > 1) or (_need > 2 and len(in_plate) < _need):
                            # recipe not satisfied
                            continue
                        if self.plate_take_requires_deliver and not any(
                                self.is_reachable(self.loc, d.location) for d in world.world_objects.get('DeliverSquare', [])):
                            # completed plate, but this seat cannot deliver it: leave it for the partner
                            continue
                        # Can't take a plate if not in position matching convention
                        if self.convention is not None and not loc in self.convention:
                            continue                    
                        # Can not take a plate with ingredients out of ingredient support set
                        out_ingredient_support_set = False
                        for ingred in in_plate:
                            if not self.in_ingredient_support(ingred):
                                out_ingredient_support_set = True
                                break
                        if out_ingredient_support_set:
                            continue
                    else:
                        # Take an ingredient. If medium or minimum policy, can't take an ingredient from the divider to somewhere else
                        # NOTE: this only works for the right hand side of the divider
                        if (self.policy_type == 'medium' or self.policy_type == 'minimum') and (loc in self.divider_loc):
                            continue
                    if self.is_reachable(self.loc, loc) and self.distance(self.loc, loc)<min_dist:
                        min_dist = self.distance(self.loc, loc)
                        e.target_location = loc
                        e.from_divider = loc in self.divider_loc
                if min_dist == inf:
                    e.available = False
                    return False
                else:
                    e.available = True
                    return True
    
    def check_still_available(self, e:event, world):
        # If current event is not done, check whether it is still available every time step
        dynamic_object = e.dynamic_obj
        static_object = e.static_obj
        target_location = e.target_location
        if e.action == "Take":
            # check whether the object is still at the target location
            if len(world.get_objects_at(target_location, StringToClass[dynamic_object]))==0:
                return False
            else:
                return True
        elif e.action == "Put":
            # check whether current object can be put to target location
            if static_object != "Plate":
                # Cannot put something on Cutboard/Divider/DeliverSquare when there is already something on it
                if len(world.get_objects_at(target_location, DynamicObject))>0:
                    return False
                else:
                    return True
            else:
                # Cannot conitune to put ingredients on a plate if the plate is gone
                if len(world.get_objects_at(target_location, Plate))==0:
                    return False
                else:
                    return True
        elif e.action == "Chop":
            # Cannot chop if the food is gone or is already chopped
            if len(world.get_objects_at(target_location, StringToClass[dynamic_object]))==0 or world.get_objects_at(target_location, StringToClass[dynamic_object])[0].done():
                return False
            else:
                return True
        else:
            raise ValueError("Unknown action")

    def get_available_events(self, world):
        return [e for e in self.support_set if self.is_event_available(e, world)]

    def select_available_event(self, world):
        cur_prob = np.zeros(len(self.support_set))
        for (i,e) in enumerate(self.support_set):
            if self.is_event_available(e, world):
                cur_prob[i] = self.event_probs[i]
        if np.sum(cur_prob)==0:
            return None
        cur_prob = cur_prob/np.sum(cur_prob) # normalization
        idx = np.random.choice(len(self.support_set),p=cur_prob)
        # print(cur_prob)
        return self.support_set[idx]

    def get_random_action(self):
        random_action = np.random.randint(5)
        #print(self.cur_event,"random action",random_action)
        return random_action

    def get_navigation_action(self, world, agent_loc, target_loc):
        # print("target location", target_loc[0])
        # print("agent_loc", agent_loc[0])
        if self.distance(agent_loc,target_loc)>1:
            # still need to move
            
            action_list = []
            if target_loc[0]>agent_loc[0] and self.is_movable(world,(agent_loc[0]+1,agent_loc[1])):
                action_list.append(2)
            elif target_loc[0]<agent_loc[0] and self.is_movable(world,(agent_loc[0]-1,agent_loc[1])):
                action_list.append(1)
            if target_loc[1]>agent_loc[1] and self.is_movable(world,(agent_loc[0],agent_loc[1]+1)):
                action_list.append(3)
            elif target_loc[1]<agent_loc[1] and self.is_movable(world,(agent_loc[0],agent_loc[1]-1)):
                action_list.append(4)
            if len(action_list)==1:
                action = action_list[0]
            elif len(action_list)>1:
                action = action_list[0] if np.random.rand()<self.nav_p else action_list[1]
            else:
                print(action_list)
                print("target location: x", target_loc[0])
                print("agent_loc: x", agent_loc[0])
                print("target location: y", target_loc[1])
                print("agent_loc: y", agent_loc[1])
                print(self.is_movable(world,(agent_loc[0]+1,agent_loc[1])))
                print(self.is_movable(world,(agent_loc[0]-1,agent_loc[1])))
                print(self.is_movable(world,(agent_loc[0],agent_loc[1]+1)))
                print(self.is_movable(world,(agent_loc[0],agent_loc[1]-1)))
                raise NotImplementedError
            
            # import sys
            # sys.exit()
        else:
            # stop and wait
            action = 0
        return action

    def get_action_from_event(self, world):
        # print('selecting the action')
        # print(self.cur_event.target_location)
        if "divider" in self.env_name:
            agent_loc = world.agents[self.agent_id].location
            agent_orientation = world.agents[self.agent_id].orientation
            target_loc = self.cur_event.target_location
            if self.distance(agent_loc,target_loc)>1:
                action = self.get_navigation_action(world, agent_loc, target_loc)
                '''
                # still need to move
                action_list = []
                if target_loc[0]>agent_loc[0] and self.is_movable(world,(agent_loc[0]+1,agent_loc[1])):
                    action_list.append(2)
                elif target_loc[0]<agent_loc[0] and self.is_movable(world,(agent_loc[0]-1,agent_loc[1])):
                    action_list.append(1)
                if target_loc[1]>agent_loc[1] and self.is_movable(world,(agent_loc[0],agent_loc[1]+1)):
                    action_list.append(3)
                elif target_loc[1]<agent_loc[1] and self.is_movable(world,(agent_loc[0],agent_loc[1]-1)):
                    action_list.append(4)
                if len(action_list)==1:
                    action = action_list[0]
        v        else:
                    action = action_list[0] if np.random.rand()<self.nav_p else action_list[1]
                '''
            else:
                # only need to change orientation if necessary
                if target_loc[0]==agent_loc[0]:
                    if target_loc[1]>agent_loc[1]:
                        orientation = 3
                    else:
                        orientation = 4
                else:
                    if target_loc[0]>agent_loc[0]:
                        orientation = 2
                    else:
                        orientation = 1
                if agent_orientation == orientation:
                    action = 5
                    self.cur_event.done = True
                else:
                    action = orientation
        else:
            raise NotImplementedError
        # with rand_p probability choose random action instead of carrying out current plan
        #print(self.cur_event, self.cur_event.target_location, action)
        if np.random.rand()<self.rand_p:
            action = self.get_random_action()
        
        return action

    def reset(self):
        pass
    
    def __call__(self, world):
        # print("Current event:",self.cur_event)
        if self.cur_event is not None:
            # judge whether the current event has become invalid
            if not self.check_still_available(self.cur_event, world):
                self.cur_event = None
        if self.cur_event is None:
            # select a new event
            new_event = self.select_available_event(world)
            # print("New event:",new_event)
            if new_event is None:
                # No available event, navigate to the plate and wait for food service
                self_loc = world.agents[self.agent_id].location
                # plate location is (4,2)
                # deliver_loc = world.world_objects['Plate'][0].location
                # print("these are all the plates in the world:", world.world_objects["Plate"])
                # print("this is the plate location:", deliver_loc)
                deliver_location = [3,2]
                # print("this is the location next to the plate on the left:", deliver_location[0])
                action = self.get_navigation_action(world, self_loc, deliver_location)
            else:
                self.cur_event = new_event 
                action = self.get_action_from_event(world)
        else:
            action = self.get_action_from_event(world)
        # print('Executing event', self.cur_event)

        if self.cur_event is not None and self.cur_event.done:
            if self.cur_event.action == "Put" and self.cur_event.static_obj == "CutBoard":
                self.cur_event = event("Chop", self.cur_event.dynamic_obj)
            elif self.cur_event.action == "Chop":
                self.cur_event = event("Take", self.cur_event.dynamic_obj)
            # elif self.cur_event.action == "Take" and self.cur_event.dynamic_obj != "Plate":s
            #     take_obj = world.get_objects_at(self.cur_event.target_location, DynamicObject)
            #     if len(take_obj) == 1 and not take_obj[0].done():
            #         self.cur_event = event("Put", self.cur_event.dynamic_obj, "CutBoard")
            #     else:
            #         self.cur_event = None
            #         self.refresh_event()
            else:
                self.cur_event = None
                self.refresh_event()        
        
        return action

        # if np.random.rand() < self.p:
        #     return self.get_random_action()
        # available_events = self.get_available_events(obs)
        # if len(available_events) == 0:
        #     return self.get_random_action()
        # if self.event_prio is not None:
        #     for e in self.event_prio:
        #         if e in available_events:
        #             selected_event = e
        #             break
        #     else:
        #         raise ValueError('ERROR: available event not found')
        # else:
        #     selected_event = np.random.choice(self.support_set, p=self.event_probs)
        # return self.get_action_from_event(selected_event)


class NullPolicy(RuleBasedPolicy):
    """Opponent that always returns the noop action (0). Used for D1 (L1 vs null partner)
    diagnostic. Subclasses RuleBasedPolicy so overcooked_single.get_oppo_action's isinstance
    check routes the world-arg call path to it.
    """
    def __init__(self, env_name='full_divider_broccoli_large_upper_random_init', ingredient_support_set=None):
        super().__init__(
            policy_type='full', nav_p=0.0, tar_p=0.0, rand_p=0.0,
            convention=None, env_name=env_name,
            ingredient_support_set=ingredient_support_set,
        )

    def __call__(self, world):
        return 0

    def reset(self):
        self.cur_event = None


from pathlib import Path
def get_train_eval_cross_seed_pool(args):
    # pick all the latest agents from logs
    train_pool = []
    eval_pool = []
    location = f"{PACE_ROOT}/logs/Overcooked/cross_seed_eval/"
    # this arg needs to be changed. Doing it manually for now.
    latent_training = True
    history_size = 5
    has_rew_done = False
    has_meta_time_step = False
    include_current_episode = True
    last_episode_only = False
    merge_encoder_computation = True
    pop_oldest_episode = False
    self_obs_mode = True
    for root, dirs,files in os.walk(location):
        for file in files:
            if file == "latest.pt":
                file_path = os.path.join(root,file)
                policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
                eval_pool.append(policy)
    
    for root, dirs,files in os.walk(location):
        for file in files:           
            if file == "25008000.pt":
                file_path = os.path.join(root,file)
                policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
                eval_pool.append(policy)
    # for root, dirs,files in os.walk(location):
    #     for file in files:           
    #         if file == "10008000.pt":
    #             file_path = os.path.join(root,file)
    #             policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
    #             eval_pool.append(policy)
    # for root, dirs,files in os.walk(location):
    #     for file in files:           
    #         if file == "15000000.pt":
    #             file_path = os.path.join(root,file)
    #             policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
    #             eval_pool.append(policy)
    # for root, dirs,files in os.walk(location):
    #     for file in files:           
    #         if file == "20004000.pt":
    #             file_path = os.path.join(root,file)
    #             policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
    #             eval_pool.append(policy)
    # for root, dirs,files in os.walk(location):
    #     for file in files:           
    #         if file == "5004000.pt":
    #             file_path = os.path.join(root,file)
    #             policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
    #             eval_pool.append(policy)
    # for root, dirs,files in os.walk(location):
    #     for file in files:           
    #         if file == "5004000.pt":
    #             file_path = os.path.join(root,file)
    #             policy = PretrainedPolicy_test(latent_training=latent_training, history_size=history_size, has_rew_done=has_rew_done, has_meta_time_step=has_meta_time_step, include_current_episode=include_current_episode, merge_encoder_computation=merge_encoder_computation, last_episode_only=last_episode_only,pop_oldest_episode=pop_oldest_episode,self_obs_mode=self_obs_mode,model_path=file_path, eval_history=False)
    #             eval_pool.append(policy)
    # eval_pool = eval_pool[0:2]
    # eval_pool = eval_pool[0:2]
    return train_pool, eval_pool

def _resolve_l1_ckpt(base, seed, preferred):
    """Find a level-1 partner checkpoint under {base}/seed{N}/ppo/ or {base}/seed{N}_seed{N}/ppo/.

    Tries the preferred file name first, then the 30M file, then latest.pt (latent
    architectures) and individual_latest.pt (joint-training architectures), so pools of
    every method load with the same loader. Returns the canonical path when nothing exists.
    """
    names = []
    for n in (preferred, '30000000.pt', 'latest.pt', 'individual_latest.pt'):
        if n and n not in names:
            names.append(n)
    for sub in (f"seed{seed}", f"seed{seed}_seed{seed}"):
        for fname in names:
            cand = os.path.join(base, sub, 'ppo', fname)
            if os.path.exists(cand):
                return cand
    return os.path.join(base, f"seed{seed}", 'ppo', names[0])


def get_train_eval_level_1_pool_custom_seeds(args, train_seeds, eval_seeds):
    """
    Load L1 agents with custom seed selection for simplified L2 training.
    
    Args:
        args: Training arguments
        train_seeds: List of seeds for training pool (e.g., [1, 4])
        eval_seeds: List of seeds for evaluation pool (e.g., [5])
    
    Returns:
        train_pool, eval_pool: Lists of loaded policies
    """
    train_pool = []
    eval_pool = []
    
    # Determine which L1 agents to load (opposite player)
    partner_side = "right" if args.player_id == 0 else "left"

    # Use new reorganized L1 agents location.
    # --l1-partner-base-dir overrides it so an L2 agent can train against a per-method
    # L1 pool (its OWN architecture), e.g. june_new_exp/baselines_FC/right/LiAM.
    override_base = getattr(args, 'l1_partner_base_dir', None)
    if override_base:
        base_location = override_base
    elif hasattr(args, 'l1_p0') and args.l1_p0:
        base_location = f"{PACE_ROOT}/logs/Overcooked/L1_agents/{partner_side}_p0"
    else:
        base_location = f"{PACE_ROOT}/logs/Overcooked/L1_agents/{partner_side}"

    # Resolve a seed dir under base_location, trying the consolidated seed{N}/ layout
    # first, then the raw seed{N}_seed{N}/ layout train_.py writes before consolidation.
    ckpt_name = getattr(args, 'l1_partner_ckpt', None) or '30000000.pt'
    def _resolve_ckpt(seed):
        return _resolve_l1_ckpt(base_location, seed, ckpt_name)

    # --feed-partner-history: same opt-in as the FI disjoint loader (see there). Default off.
    feed_kwargs = {}
    if getattr(args, 'feed_partner_history', False):
        import yaml as _yaml
        with open(args.env_config) as _f:
            _horizon = _yaml.safe_load(_f).get('horizon')
        feed_kwargs = {'feed_history': True, 'max_episode_length': int(_horizon)}

    def _apply_feed_mode(policy):
        # Partners share the L2's architecture flags (per-method regime), so the L2's args say how the
        # partner's own training filled its memory: own vs other action (LiLi) and meta-episode GRU state.
        if feed_kwargs:
            policy.self_action_mode = bool(getattr(args, 'self_action_mode', False))
            policy.use_meta_episode = bool(getattr(args, 'use_meta_episode', False))
        return policy

    print(f'Loading L1 {partner_side} agents with CUSTOM SEEDS for L2 (player_id={args.player_id})')
    print(f'  Training seeds: {train_seeds}')
    print(f'  Evaluation seeds: {eval_seeds}')
    print(f'  Base location: {base_location}')
    print(f'  Checkpoint file: {ckpt_name}')
    print('  Partner history: ' + ('FED during training' if feed_kwargs else 'not written (default)'))

    # Load training pool
    for seed in train_seeds:
        checkpoint_path = _resolve_ckpt(seed)
        if os.path.exists(checkpoint_path):
            policy = PretrainedPolicy_test(
                latent_training=args.latent_training,
                history_size=args.history_size,
                has_rew_done=args.has_rew_done,
                has_meta_time_step=args.has_meta_time_step,
                include_current_episode=args.include_current_episode,
                merge_encoder_computation=args.merge_encoder_computation,
                last_episode_only=args.last_episode_only,
                pop_oldest_episode=args.pop_oldest_episode,
                self_obs_mode=args.self_obs_mode,
                model_path=checkpoint_path,
                eval_history=True,
                agent_id=1-args.player_id,
                **feed_kwargs,
            )
            train_pool.append(_apply_feed_mode(policy))
            print(f'  ✓ Train pool: Loaded seed {seed} ({checkpoint_path})')
        else:
            print(f'  ✗ ERROR: Training checkpoint not found: {checkpoint_path}')
            raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")

    # Load evaluation pool
    for seed in eval_seeds:
        checkpoint_path = _resolve_ckpt(seed)
        if os.path.exists(checkpoint_path):
            policy = PretrainedPolicy_test(
                latent_training=args.latent_training,
                history_size=args.history_size,
                has_rew_done=args.has_rew_done,
                has_meta_time_step=args.has_meta_time_step,
                include_current_episode=args.include_current_episode,
                merge_encoder_computation=args.merge_encoder_computation,
                last_episode_only=args.last_episode_only,
                pop_oldest_episode=args.pop_oldest_episode,
                self_obs_mode=args.self_obs_mode,
                model_path=checkpoint_path,
                eval_history=True,
                agent_id=1-args.player_id,
                **feed_kwargs,
            )
            eval_pool.append(_apply_feed_mode(policy))
            print(f'  ✓ Eval pool: Loaded seed {seed} ({checkpoint_path})')
        else:
            print(f'  ✗ ERROR: Evaluation checkpoint not found: {checkpoint_path}')
            raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
    
    print(f'Successfully loaded {len(train_pool)} training and {len(eval_pool)} evaluation L1 {partner_side} agents')
    
    return train_pool, eval_pool


def get_train_eval_mixed_L1_L0_pool(args, train_seeds, eval_seeds=None, num_l0_train=3, num_l0_eval=3):
    """
    Load a MIXED pool of L1 agents + L0 rule-based agents for L2 training.
    
    This helps L2 learn to:
    - Adapt to learning L1 partners (who may have different conventions)
    - Adapt to deterministic L0 partners (forces clear convention following)
    - Take initiative OR follow, depending on partner behavior
    
    Args:
        args: Training arguments
        train_seeds: List of L1 seeds for training pool
        eval_seeds: List of L1 seeds for eval pool (optional, if None only L0 for eval)
        num_l0_train: Number of L0 rule-based agents for training (default: 3, one per convention)
        num_l0_eval: Number of L0 rule-based agents for evaluation (default: 3)
    
    Returns:
        train_pool, eval_pool: Lists of mixed policies
    """
    import yaml
    
    train_pool = []
    eval_pool = []
    
    # Load L1 agents first
    partner_side = "right" if args.player_id == 0 else "left"
    
    if hasattr(args, 'l1_p0') and args.l1_p0:
        base_location = f"{PACE_ROOT}/logs/Overcooked/L1_agents/{partner_side}_p0"
    elif hasattr(args, 'l1_p025') and args.l1_p025:
        base_location = f"{PACE_ROOT}/logs/Overcooked/june_new_exp/L1_FC/{partner_side}"
    else:
        base_location = f"{PACE_ROOT}/logs/Overcooked/L1_agents/{partner_side}"
    
    print(f'='*70)
    print(f'Loading MIXED L1+L0 pool for L2 training')
    print(f'='*70)
    print(f'L1 {partner_side} agents: train seeds {train_seeds}, eval seeds {eval_seeds}')
    print(f'L0 rule-based agents: {num_l0_train} for training, {num_l0_eval} for eval')
    print(f'Base location: {base_location}')
    print()
    
    # Load L1 training pool
    print('Loading L1 agents for training pool:')
    for seed in train_seeds:
        checkpoint_path = f"{base_location}/seed{seed}/ppo/30000000.pt"
        if os.path.exists(checkpoint_path):
            policy = PretrainedPolicy_test(
                latent_training=args.latent_training,
                history_size=args.history_size,
                has_rew_done=args.has_rew_done,
                has_meta_time_step=args.has_meta_time_step,
                include_current_episode=args.include_current_episode,
                merge_encoder_computation=args.merge_encoder_computation,
                last_episode_only=args.last_episode_only,
                pop_oldest_episode=args.pop_oldest_episode,
                self_obs_mode=args.self_obs_mode,
                model_path=checkpoint_path,
                eval_history=True,
                agent_id=1-args.player_id
            )
            train_pool.append(policy)
            print(f'  ✓ L1 seed {seed}')
        else:
            print(f'  ✗ ERROR: L1 checkpoint not found: {checkpoint_path}')
            raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
    
    # Load L0 rule-based agents for training pool
    print(f'\nLoading L0 rule-based agents for training pool:')
    with open(args.env_config, 'r') as env_config_file:
        env_map = yaml.safe_load(env_config_file)['mode']
    
    # Generate L0 policies (one per convention: LettuceOnion, PotatoBroccoli, TomatoCarrot)
    l0_train_policies = generate_policy_pool(
        gen_desire=False, 
        p_max=args.p,  # Use args.p for randomness (0.0 for deterministic)
        env_name=env_map,
        pool_size=num_l0_train,
        recipe_type=args.recipe_type,
        pool_seed=1
    )
    
    for i, l0_pol in enumerate(l0_train_policies):
        train_pool.append(l0_pol)
        print(f'  ✓ L0 convention {i}: {l0_pol.ingredient_support_set}')
    
    # Load L1 agents for evaluation pool (if eval_seeds provided)
    if eval_seeds is not None and len(eval_seeds) > 0:
        print(f'\nLoading L1 agents for evaluation pool:')
        for seed in eval_seeds:
            checkpoint_path = f"{base_location}/seed{seed}/ppo/30000000.pt"
            if os.path.exists(checkpoint_path):
                policy = PretrainedPolicy_test(
                    latent_training=args.latent_training,
                    history_size=args.history_size,
                    has_rew_done=args.has_rew_done,
                    has_meta_time_step=args.has_meta_time_step,
                    include_current_episode=args.include_current_episode,
                    merge_encoder_computation=args.merge_encoder_computation,
                    last_episode_only=args.last_episode_only,
                    pop_oldest_episode=args.pop_oldest_episode,
                    self_obs_mode=args.self_obs_mode,
                    model_path=checkpoint_path,
                    eval_history=True,
                    agent_id=1-args.player_id
                )
                eval_pool.append(policy)
                print(f'  ✓ L1 eval seed {seed}')
            else:
                print(f'  ✗ WARNING: L1 eval checkpoint not found: {checkpoint_path}')
    
    # Load L0 rule-based agents for evaluation pool
    print(f'\nLoading L0 rule-based agents for evaluation pool:')
    l0_eval_policies = generate_policy_pool(
        gen_desire=False,
        p_max=args.p,
        env_name=env_map,
        pool_size=num_l0_eval,
        recipe_type=args.recipe_type,
        pool_seed=2  # Different seed for eval
    )
    
    for i, l0_pol in enumerate(l0_eval_policies):
        eval_pool.append(l0_pol)
        print(f'  ✓ L0 eval convention {i}: {l0_pol.ingredient_support_set}')
    
    num_l1_eval = len(eval_seeds) if eval_seeds else 0
    print()
    print(f'='*70)
    print(f'MIXED POOL SUMMARY:')
    print(f'  Training: {len(train_pool)} total ({len(train_seeds)} L1 + {num_l0_train} L0)')
    print(f'  Evaluation: {len(eval_pool)} total ({num_l1_eval} L1 + {num_l0_eval} L0)')
    print(f'='*70)
    
    return train_pool, eval_pool


def get_train_eval_level_1_pool(args):
    """
    Load L1 agents as partners for L2 training.
    
    For L2 LEFT (player_id=0): loads L1 RIGHT agents (player_id=1) as partners
    For L2 RIGHT (player_id=1): loads L1 LEFT agents (player_id=0) as partners
    
    Training pool: by default loads L1 agents from seeds 1..train_pool_size (latest.pt)
    Evaluation pool: by default loads L1 agents from seeds train_pool_size+1 .. train_pool_size+eval_pool_size (latest.pt)
    
    If args.l1_eval_seeds is set (comma-separated string like "5,10,2"), use those specific seeds for evaluation.
    """
    train_pool = []
    eval_pool = []
    
    # Determine which L1 agents to load (opposite player)
    partner_side = "right" if args.player_id == 0 else "left"
    base_location = f"{PACE_ROOT}/logs/Overcooked/june_new_exp/L1_FC/{partner_side}"
    
    print(f'Loading L1 {partner_side} agents as partners for L2 (player_id={args.player_id})')
    
    # Check if attributes exist (L2 agents have them, baselines may not)
    latent_training = getattr(args, 'latent_training', False)
    history_size = getattr(args, 'history_size', 5)
    has_rew_done = getattr(args, 'has_rew_done', False)
    has_meta_time_step = getattr(args, 'has_meta_time_step', False)
    include_current_episode = getattr(args, 'include_current_episode', False)
    merge_encoder_computation = getattr(args, 'merge_encoder_computation', False)
    last_episode_only = getattr(args, 'last_episode_only', False)
    pop_oldest_episode = getattr(args, 'pop_oldest_episode', False)
    self_obs_mode = getattr(args, 'self_obs_mode', False)
    
    # Check for custom eval seeds
    eval_seeds_custom = None
    if hasattr(args, 'l1_eval_seeds') and args.l1_eval_seeds:
        # Parse comma-separated seed list
        eval_seeds_custom = [int(s.strip()) for s in args.l1_eval_seeds.split(',')]
        print(f'  Using CUSTOM evaluation seeds: {eval_seeds_custom}')
    
    # Training pool: seeds 1..train_pool_size
    for seed in range(1, args.train_pool_size + 1):
        checkpoint_path = f"{base_location}/seed{seed}/ppo/latest.pt"
        if os.path.exists(checkpoint_path):
            policy = PretrainedPolicy_test(
                latent_training=latent_training,
                history_size=history_size,
                has_rew_done=has_rew_done,
                has_meta_time_step=has_meta_time_step,
                include_current_episode=include_current_episode,
                merge_encoder_computation=merge_encoder_computation,
                last_episode_only=last_episode_only,
                pop_oldest_episode=pop_oldest_episode,
                self_obs_mode=self_obs_mode,
                model_path=checkpoint_path,
                eval_history=False,
                agent_id=1-args.player_id
            )
            train_pool.append(policy)
            print(f'  Train pool: Loaded {checkpoint_path}')
        else:
            print(f'  WARNING: Training checkpoint not found: {checkpoint_path}')

    # Evaluation pool: use custom seeds if specified, otherwise default sequential seeds
    if eval_seeds_custom:
        eval_seed_list = eval_seeds_custom
    else:
        start_eval_seed = args.train_pool_size + 1
        end_eval_seed = args.train_pool_size + args.eval_pool_size
        eval_seed_list = range(start_eval_seed, end_eval_seed + 1)
    
    for seed in eval_seed_list:
        checkpoint_path = f"{base_location}/seed{seed}/ppo/latest.pt"
        if os.path.exists(checkpoint_path):
            policy = PretrainedPolicy_test(
                latent_training=latent_training,
                history_size=history_size,
                has_rew_done=has_rew_done,
                has_meta_time_step=has_meta_time_step,
                include_current_episode=include_current_episode,
                merge_encoder_computation=merge_encoder_computation,
                last_episode_only=last_episode_only,
                pop_oldest_episode=pop_oldest_episode,
                self_obs_mode=self_obs_mode,
                model_path=checkpoint_path,
                eval_history=False,
                agent_id=1-args.player_id
            )
            eval_pool.append(policy)
            print(f'  Eval pool: Loaded {checkpoint_path}')
        else:
            print(f'  WARNING: Evaluation checkpoint not found: {checkpoint_path}')
    
    print(f'Loaded {len(train_pool)} L1 {partner_side} agents for training')
    print(f'Loaded {len(eval_pool)} L1 {partner_side} agents for evaluation')
    
    return train_pool, eval_pool


def get_train_eval_specialized_L1_pool(args, train_spec, eval_spec):
    """
    Load SPECIALIZED L1 agents (trained on recipe subsets) for L2 training.
    
    Args:
        args: Training arguments
        train_spec: String like "LO_PB:1,2,3;LO_TC:1,2,3;PB_TC:1,2,3"
                    Format: TYPE:seed1,seed2,...;TYPE:seed1,seed2,...
        eval_spec: String like "LO_PB:4,5;LO_TC:4,5;PB_TC:4,5"
    
    Example:
        train_spec = "LO_PB:1,2,3;LO_TC:1,2,3;PB_TC:1,2,3"
        Loads 9 agents: LO_PB seeds 1,2,3 + LO_TC seeds 1,2,3 + PB_TC seeds 1,2,3
    
    Returns:
        train_pool, eval_pool: Lists of PretrainedPolicy_test objects
    """
    train_pool = []
    eval_pool = []
    
    partner_side = "right" if args.player_id == 0 else "left"
    # Default is the original ppo_L1_specialized pool; --l1-specialized-base-dir overrides
    # it (e.g. to point at the peer-id pool logs/Overcooked/june_new_exp/L1_PC_p05_peerid/right).
    base_location = getattr(args, 'l1_specialized_base_dir', None) or \
        f"{PACE_ROOT}/logs/Overcooked/ppo_L1_specialized"
    
    print(f'='*70)
    print(f'Loading SPECIALIZED L1 pool for L2 training')
    print(f'='*70)
    print(f'L1 specialized agents (player side: {partner_side})')
    print(f'Base location: {base_location}')
    print()
    
    # Parse training spec
    print('Loading L1 SPECIALIZED agents for training pool:')
    for type_spec in train_spec.split(';'):
        agent_type, seeds_str = type_spec.strip().split(':')
        seeds = [int(s.strip()) for s in seeds_str.split(',')]
        
        for seed in seeds:
            # Consolidated layout: {base}/{TYPE}/seed{N}/ppo/latest.pt
            # Raw layout:          {base}/{TYPE}/seed{N}_seed{N}/ppo/latest.pt
            checkpoint_path = _resolve_l1_ckpt(f"{base_location}/{agent_type}", seed, 'latest.pt')

            if os.path.exists(checkpoint_path):
                policy = PretrainedPolicy_test(
                    latent_training=args.latent_training,
                    history_size=args.history_size,
                    has_rew_done=args.has_rew_done,
                    has_meta_time_step=args.has_meta_time_step,
                    include_current_episode=args.include_current_episode,
                    merge_encoder_computation=args.merge_encoder_computation,
                    last_episode_only=args.last_episode_only,
                    pop_oldest_episode=args.pop_oldest_episode,
                    self_obs_mode=args.self_obs_mode,
                    model_path=checkpoint_path,
                    eval_history=True,
                    agent_id=1-args.player_id
                )
                train_pool.append(policy)
                print(f'  ✓ {agent_type} seed {seed}')
            else:
                print(f'  ✗ ERROR: Checkpoint not found: {checkpoint_path}')
                raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")
    
    # Parse eval spec
    print(f'\nLoading L1 SPECIALIZED agents for evaluation pool:')
    for type_spec in eval_spec.split(';'):
        agent_type, seeds_str = type_spec.strip().split(':')
        seeds = [int(s.strip()) for s in seeds_str.split(',')]
        
        for seed in seeds:
            checkpoint_path = _resolve_l1_ckpt(f"{base_location}/{agent_type}", seed, 'latest.pt')

            if os.path.exists(checkpoint_path):
                policy = PretrainedPolicy_test(
                    latent_training=args.latent_training,
                    history_size=args.history_size,
                    has_rew_done=args.has_rew_done,
                    has_meta_time_step=args.has_meta_time_step,
                    include_current_episode=args.include_current_episode,
                    merge_encoder_computation=args.merge_encoder_computation,
                    last_episode_only=args.last_episode_only,
                    pop_oldest_episode=args.pop_oldest_episode,
                    self_obs_mode=args.self_obs_mode,
                    model_path=checkpoint_path,
                    eval_history=True,
                    agent_id=1-args.player_id
                )
                eval_pool.append(policy)
                print(f'  ✓ {agent_type} seed {seed}')
            else:
                print(f'  ✗ WARNING: Eval checkpoint not found: {checkpoint_path}')
    
    print()
    print(f'='*70)
    print(f'SPECIALIZED L1 POOL SUMMARY:')
    print(f'  Training: {len(train_pool)} specialized L1 agents')
    print(f'  Evaluation: {len(eval_pool)} specialized L1 agents')
    print(f'='*70)
    
    return train_pool, eval_pool


class RandomizedTypePolicy:
    """Wraps N seeds of the same L1 type. On each reset(), picks one seed uniformly."""
    def __init__(self, seed_policies):
        self._policies = seed_policies
        self._active = seed_policies[0]

    def reset(self):
        import random
        self._active = random.choice(self._policies)
        self._active.reset()

    def act(self, obs, rnn_states, masks, deterministic=False):
        return self._active.act(obs, rnn_states, masks, deterministic)

    def __call__(self, *args, **kwargs):
        return self._active(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_active'), name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_active'), name, value)


class CurriculumPool:
    """
    Curriculum-aware opponent pool.

    On reset(), samples from L0 with probability (1 - l1_prob) and from L1 with
    probability l1_prob, where l1_prob ramps linearly:
      0           → warmup_steps:              pure L0 (l1_prob = 0)
      warmup      → warmup + ramp:             linear ramp   (l1_prob 0 → 1)
      warmup+ramp → end:                       pure L1 (l1_prob = 1)

    Each process estimates total training steps as:
        local_episode_count * steps_per_episode * num_processes
    (approximate; fine for a smooth curriculum schedule).
    """
    def __init__(self, l0_policies, l1_policies,
                 warmup_steps, ramp_steps,
                 steps_per_episode=1000, num_processes=15):
        self._l0 = l0_policies
        self._l1 = l1_policies
        self._warmup = warmup_steps
        self._ramp = ramp_steps
        self._steps_per_ep = steps_per_episode
        self._num_proc = num_processes
        self._episode_count = 0
        self._opponent_id = None
        self._active = l0_policies[0]

    def set_id(self, agent_id):
        self._opponent_id = agent_id
        self._active.set_id(agent_id)

    def _l1_prob(self):
        import random as _random
        total_steps = self._episode_count * self._steps_per_ep * self._num_proc
        if total_steps < self._warmup:
            return 0.0
        return min(1.0, (total_steps - self._warmup) / self._ramp)

    def reset(self):
        import random as _random
        self._episode_count += 1
        if _random.random() < self._l1_prob():
            self._active = _random.choice(self._l1)
        else:
            self._active = _random.choice(self._l0)
        if self._opponent_id is not None and hasattr(self._active, 'set_id'):
            self._active.set_id(self._opponent_id)
        self._active.reset()

    def act(self, obs, rnn_states, masks, deterministic=False):
        return self._active.act(obs, rnn_states, masks, deterministic)

    def __call__(self, *args, **kwargs):
        return self._active(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_active'), name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_active'), name, value)


def get_train_eval_curriculum_pool(args, train_spec, eval_spec,
                                    num_l0_train=3, num_l0_eval=3,
                                    warmup_steps=5_000_000,
                                    ramp_steps=25_000_000):
    """
    Curriculum version of get_train_eval_specialized_L1_L0_pool.

    Returns train_pool of size 1 — a single CurriculumPool that starts sampling
    only L0 partners and linearly ramps to L1-only over training.

    Eval pool is the standard 6-slot pool (3 L1 eval seeds + 3 L0) so that
    evaluation numbers are comparable to Setting A / Setting B runs.

    Default schedule (60M total steps):
      0  → 5M:    pure L0   (8% of training — lets L2 learn basic cooking)
      5M → 30M:   linear ramp L0 → L1
      30M → 60M:  pure L1   (50% of training for L1 adaptation refinement)
    """
    import yaml
    import random as _random

    partner_side = "right" if args.player_id == 0 else "left"

    def _ckpt(agent_type, seed):
        if partner_side == "right":
            return (f"{PACE_ROOT}/logs/Overcooked"
                    f"/ppo_L1_specialized/{agent_type}/seed{seed}_seed{seed}/ppo/latest.pt")
        else:
            return (f"{PACE_ROOT}/logs/Overcooked"
                    f"/june_new_exp/L1_PC/left/{agent_type}/seed{seed}/ppo/latest.pt")

    def _load(ckpt):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"L1_PC checkpoint not found: {ckpt}")
        return PretrainedPolicy_test(
            latent_training=args.latent_training,
            history_size=args.history_size,
            has_rew_done=args.has_rew_done,
            has_meta_time_step=args.has_meta_time_step,
            include_current_episode=args.include_current_episode,
            merge_encoder_computation=args.merge_encoder_computation,
            last_episode_only=args.last_episode_only,
            pop_oldest_episode=args.pop_oldest_episode,
            self_obs_mode=args.self_obs_mode,
            model_path=ckpt,
            eval_history=True,
            agent_id=1 - args.player_id,
        )

    def _parse_spec(spec):
        result = {}
        for type_spec in spec.split(';'):
            agent_type, seeds_str = type_spec.strip().split(':')
            result[agent_type] = [int(s.strip()) for s in seeds_str.split(',')]
        return result

    print('=' * 70)
    print(f'Loading CURRICULUM pool — partner side: {partner_side}')
    print(f'  Warmup: {warmup_steps:,} steps (pure L0)')
    print(f'  Ramp:   {warmup_steps:,} → {warmup_steps + ramp_steps:,} steps (L0 → L1)')
    print(f'  Full L1: {warmup_steps + ramp_steps:,}+ steps')
    print('=' * 70)

    with open(args.env_config, 'r') as f:
        env_map = yaml.safe_load(f)['mode']

    # --- L1 policies for curriculum (individual seeds, not type-pooled) ---
    print('\nLoading L1_PC agents (curriculum train pool):')
    l1_policies = []
    train_types = _parse_spec(train_spec)
    for agent_type, seeds in train_types.items():
        for seed in seeds:
            ckpt = _ckpt(agent_type, seed)
            pol = _load(ckpt)
            l1_policies.append(pol)
            print(f'  ✓ {agent_type} seed {seed}')

    # --- L0 policies for curriculum ---
    print(f'\nLoading {num_l0_train} L0 rule-based agents (curriculum train pool):')
    l0_policies = generate_policy_pool(
        gen_desire=False, p_max=args.p, env_name=env_map,
        pool_size=num_l0_train, recipe_type=args.recipe_type, pool_seed=1,
    )
    for i, pol in enumerate(l0_policies):
        print(f'  ✓ L0 convention {i}: {pol.ingredient_support_set}')

    # Build single CurriculumPool
    curriculum_pool = CurriculumPool(
        l0_policies=l0_policies,
        l1_policies=l1_policies,
        warmup_steps=warmup_steps,
        ramp_steps=ramp_steps,
        steps_per_episode=args.num_steps,
        num_processes=args.num_processes,
    )
    train_pool = [curriculum_pool]

    # --- eval pool: standard 3 L1 eval seeds + 3 L0 (same as Setting A) ---
    print('\nLoading L1_PC agents for eval pool:')
    eval_pool = []
    eval_types = _parse_spec(eval_spec)
    for agent_type, seeds in eval_types.items():
        for seed in seeds:
            ckpt = _ckpt(agent_type, seed)
            try:
                pol = _load(ckpt)
                eval_pool.append(pol)
                print(f'  ✓ {agent_type} seed {seed}')
            except FileNotFoundError:
                print(f'  ✗ WARNING: {agent_type} seed {seed} not found, skipping')

    print(f'\nLoading {num_l0_eval} L0 rule-based agents for eval pool:')
    l0_eval = generate_policy_pool(
        gen_desire=False, p_max=args.p, env_name=env_map,
        pool_size=num_l0_eval, recipe_type=args.recipe_type, pool_seed=2,
    )
    for i, pol in enumerate(l0_eval):
        eval_pool.append(pol)
        print(f'  ✓ L0 eval convention {i}: {pol.ingredient_support_set}')

    print()
    print('=' * 70)
    print('CURRICULUM POOL SUMMARY:')
    print(f'  Train pool: 1 CurriculumPool ({len(l0_policies)} L0 + {len(l1_policies)} L1 internally)')
    print(f'  Eval pool:  {len(eval_pool)} partners')
    print('=' * 70)

    return train_pool, eval_pool


def get_train_eval_specialized_L1_L0_pool(args, train_spec, eval_spec,
                                           num_l0_train=3, num_l0_eval=3,
                                           type_sampling=False):
    """
    Mixed pool of SPECIALIZED L1_PC agents + L0 rule-based agents.

    Two modes (type_sampling flag):
      Setting A (type_sampling=False): each seed is an individual partner.
        train_pool = 9 L1 seeds + 3 L0 = 12 total.  Uniform over 12.
      Setting B (type_sampling=True): each type becomes one RandomizedTypePolicy.
        train_pool = 3 type-slots + 3 L0 = 6 total.  Uniform over 6;
        within each type-slot a seed is picked randomly per episode.

    Path resolution (mirrors left/right split):
      player_id=0 (LEFT L2) → partner_side=right → ppo_L1_specialized/{type}/seed{N}_seed{N}/ppo/latest.pt
      player_id=1 (RIGHT L2) → partner_side=left  → june_new_exp/L1_PC/left/{type}/seed{N}/ppo/latest.pt
    """
    import yaml

    partner_side = "right" if args.player_id == 0 else "left"

    def _ckpt(agent_type, seed):
        if partner_side == "right":
            return (f"{PACE_ROOT}/logs/Overcooked"
                    f"/ppo_L1_specialized/{agent_type}/seed{seed}_seed{seed}/ppo/latest.pt")
        else:
            return (f"{PACE_ROOT}/logs/Overcooked"
                    f"/june_new_exp/L1_PC/left/{agent_type}/seed{seed}/ppo/latest.pt")

    def _load(ckpt):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"L1_PC checkpoint not found: {ckpt}")
        return PretrainedPolicy_test(
            latent_training=args.latent_training,
            history_size=args.history_size,
            has_rew_done=args.has_rew_done,
            has_meta_time_step=args.has_meta_time_step,
            include_current_episode=args.include_current_episode,
            merge_encoder_computation=args.merge_encoder_computation,
            last_episode_only=args.last_episode_only,
            pop_oldest_episode=args.pop_oldest_episode,
            self_obs_mode=args.self_obs_mode,
            model_path=ckpt,
            eval_history=True,
            agent_id=1 - args.player_id,
        )

    def _parse_spec(spec):
        result = {}
        for type_spec in spec.split(';'):
            agent_type, seeds_str = type_spec.strip().split(':')
            result[agent_type] = [int(s.strip()) for s in seeds_str.split(',')]
        return result

    mode = "Setting B (type-sampling)" if type_sampling else "Setting A (individual seeds)"
    print('=' * 70)
    print(f'Loading SPECIALIZED L1_PC + L0 mixed pool — {mode}')
    print(f'Partner side: {partner_side}')
    print('=' * 70)

    with open(args.env_config, 'r') as f:
        env_map = yaml.safe_load(f)['mode']

    train_pool = []
    eval_pool  = []

    # --- training pool: L1_PC ---
    print('\nLoading L1_PC agents for training pool:')
    train_types = _parse_spec(train_spec)
    for agent_type, seeds in train_types.items():
        type_policies = []
        for seed in seeds:
            ckpt = _ckpt(agent_type, seed)
            pol = _load(ckpt)
            type_policies.append(pol)
            print(f'  ✓ {agent_type} seed {seed}')
        if type_sampling:
            train_pool.append(RandomizedTypePolicy(type_policies))
            print(f'  → wrapped as RandomizedTypePolicy ({len(type_policies)} seeds)')
        else:
            train_pool.extend(type_policies)

    # --- training pool: L0 ---
    print(f'\nLoading {num_l0_train} L0 rule-based agents for training pool:')
    l0_train = generate_policy_pool(
        gen_desire=False, p_max=args.p, env_name=env_map,
        pool_size=num_l0_train, recipe_type=args.recipe_type, pool_seed=1,
    )
    for i, pol in enumerate(l0_train):
        train_pool.append(pol)
        print(f'  ✓ L0 convention {i}: {pol.ingredient_support_set}')

    # --- eval pool: L1_PC ---
    print('\nLoading L1_PC agents for eval pool:')
    eval_types = _parse_spec(eval_spec)
    for agent_type, seeds in eval_types.items():
        type_policies = []
        for seed in seeds:
            ckpt = _ckpt(agent_type, seed)
            try:
                pol = _load(ckpt)
                type_policies.append(pol)
                print(f'  ✓ {agent_type} seed {seed}')
            except FileNotFoundError:
                print(f'  ✗ WARNING: {agent_type} seed {seed} not found, skipping')
        if type_policies:
            if type_sampling:
                eval_pool.append(RandomizedTypePolicy(type_policies))
            else:
                eval_pool.extend(type_policies)

    # --- eval pool: L0 ---
    print(f'\nLoading {num_l0_eval} L0 rule-based agents for eval pool:')
    l0_eval = generate_policy_pool(
        gen_desire=False, p_max=args.p, env_name=env_map,
        pool_size=num_l0_eval, recipe_type=args.recipe_type, pool_seed=2,
    )
    for i, pol in enumerate(l0_eval):
        eval_pool.append(pol)
        print(f'  ✓ L0 eval convention {i}: {pol.ingredient_support_set}')

    print()
    print('=' * 70)
    print('SPECIALIZED L1_PC + L0 POOL SUMMARY:')
    print(f'  Training: {len(train_pool)} partners ({"type-slots" if type_sampling else "individual seeds"})')
    print(f'  Evaluation: {len(eval_pool)} partners')
    print('=' * 70)

    return train_pool, eval_pool


def get_train_eval_pool(args):
    assert args.env_name == 'Overcooked'
    left_ingred = ingredients[:3]
    right_ingred = ingredients[3:]
    ingredient_sets_all = []
    policy_pool_train = []
    policy_pool_eval = []
    with open(args.env_config, 'r') as env_config_file:
        env_map = yaml.safe_load(env_config_file)['mode']
        print('Using map', env_map, 'and recipe type', args.recipe_type)
    # print(left_ingred[3])
    i=0
    while i<len(left_ingred):
        print(i)
        # print(left_ingred[i], right_ingred[i])
        ingredient_sets_all.append([left_ingred[i], right_ingred[i]])
        i = i + 1
    print(ingredient_sets_all)
    
    # Use pool_seed to shuffle recipe order, then select train_pool_size recipes
    old_state = np.random.get_state()
    np.random.seed(args.pool_seed)
    shuffled_indices = np.arange(len(ingredient_sets_all))
    np.random.shuffle(shuffled_indices)
    np.random.set_state(old_state)
    
    # Select recipes based on pool sizes
    ingredient_set_train = [ingredient_sets_all[shuffled_indices[i % len(ingredient_sets_all)]] 
                           for i in range(args.train_pool_size)]
    ingredient_set_eval = [ingredient_sets_all[shuffled_indices[i % len(ingredient_sets_all)]] 
                          for i in range(args.eval_pool_size)]
    print(f'Selected train recipes (pool_seed={args.pool_seed}): {ingredient_set_train}')
    print(f'Selected eval recipes (pool_seed={args.pool_seed}): {ingredient_set_eval}')
    
    p_max = args.p
    for recipe in ingredient_set_train:
        policy = RuleBasedPolicy('full', np.random.rand() * p_max, 0, 0.1 * np.random.rand() * p_max, None,
                                 env_map, ingredient_support_set=recipe)
        policy_pool_train.append(policy)
    
    # Use eval_pool_size from command-line arguments
    for recipe in ingredient_set_eval:
        policy = RuleBasedPolicy('full', np.random.rand() * p_max, 0, 0.1 * np.random.rand() * p_max, None,
                                 env_map, ingredient_support_set=recipe)
        policy_pool_eval.append(policy)
        
    print('Put', len(policy_pool_train), 'rule-based opponents into train pool, ingredient support sets:',
              [p.ingredient_support_set for p in policy_pool_train])
    print('Put', len(policy_pool_eval), 'rule-based opponents into eval pool, ingredient support sets:',
              [p.ingredient_support_set for p in policy_pool_eval])
    self_play_opponents = args.train_pool_size - args.rule_based_opponents
    if self_play_opponents > 0:
        assert 'potato_hard' in args.env_config, f'Loading potato hard fcp checkpoints for map {args.env_config}'
        self_play_pool = load_potato_hard_self_play_policy_pool(1 - args.player_id)
        assert len(self_play_pool) >= self_play_opponents, \
            f'Requesting {self_play_opponents} self-play opponents, got {len(self_play_pool)}'
        policy_pool_train.extend(self_play_pool[:self_play_opponents])
        print('Put', self_play_opponents, 'self-play opponents into train pool, model paths:',
              [p.model_path for p in self_play_pool[:self_play_opponents]])
    return policy_pool_train, policy_pool_eval

def get_train_eval_base_pool(args):
    assert args.env_name == 'Overcooked'
    if args.desire_id is not None:
        assert args.desire_id < 2 ** 5, f'Desire id out of range: {args.desire_id}'
        policy_pool_train = [[((args.desire_id >> i) & 1) for i in range(5)]]
        policy_pool_eval = []
        self_play_opponents = 0
        print('Put 1 desire into train pool')
    elif args.rule_based_opponents > 0 or args.eval_pool_size > 0:
        with open(args.env_config, 'r') as env_config_file:
            env_map = yaml.safe_load(env_config_file)['mode']
        print('Using map', env_map, 'and recipe type', args.recipe_type)
        policy_pool_train_eval = generate_policy_pool(args.multi_agent > 1, args.p, env_map,
                                                      args.rule_based_opponents + args.eval_pool_size,
                                                      args.recipe_type, args.pool_seed)
        policy_pool_train = policy_pool_train_eval[:args.rule_based_opponents]
        policy_pool_eval = policy_pool_train_eval[args.rule_based_opponents:]
        self_play_opponents = args.train_pool_size - args.rule_based_opponents
        print('Put', len(policy_pool_train), 'rule-based opponents into train pool, ingredient support sets:',
              [p.ingredient_support_set for p in policy_pool_train])
        print('Put', len(policy_pool_eval), 'rule-based opponents into eval pool, ingredient support sets:',
              [p.ingredient_support_set for p in policy_pool_eval])
    else:
        policy_pool_train = []
        policy_pool_eval = []
        self_play_opponents = args.train_pool_size
    if self_play_opponents > 0:
        assert 'potato_hard' in args.env_config, f'Loading potato hard fcp checkpoints for map {args.env_config}'
        self_play_pool = load_potato_hard_self_play_policy_pool(1 - args.player_id)
        assert len(self_play_pool) >= self_play_opponents, \
            f'Requesting {self_play_opponents} self-play opponents, got {len(self_play_pool)}'
        policy_pool_train.extend(self_play_pool[:self_play_opponents])
        print('Put', self_play_opponents, 'self-play opponents into train pool, model paths:',
              [p.model_path for p in self_play_pool[:self_play_opponents]])
    return policy_pool_train, policy_pool_eval


def generate_policy_pool(gen_desire, p_max, env_name, pool_size, recipe_type, pool_seed=1):
    old_state = np.random.get_state()
    np.random.seed(pool_seed)
    left_ingred = ingredients[:3]
    right_ingred = ingredients[3:]
    ingredient_sets_all = []
    # for i in range(1, 2 ** len(ingredients)):
    #     if i > (i & -i):
    #         ingredient_sets_all.append([ingredients[j] for j in range(len(ingredients)) if (i >> j) & 1])

    # if recipe_type == 'full':
    #     # one ingredient in left
    #     for ingred in left_ingred:
    #         ingredient_sets_all.append([ingred])
    #     # two ingredient in left
    #     for i in range(len(left_ingred)):
    #         for j in range(i+1, len(left_ingred)):
    #             ingredient_sets_all.append([left_ingred[i], left_ingred[j]])
    # else:
    #     assert recipe_type == 'cross', f'Unknown recipe type {recipe_type}'

    # # one ingredient in left, one in right
    # for ingred in left_ingred:
    #     for ingred2 in right_ingred:
    #         ingredient_sets_all.append([ingred, ingred2])
    i=0
    while i<len(left_ingred):
        print(i)
        # print(left_ingred[i], right_ingred[i])
        ingredient_sets_all.append([left_ingred[i], right_ingred[i]])
        i = i + 1
    print(ingredient_sets_all)
    print('All ingredient support sets:', ingredient_sets_all)
    print('Corresponding indices:', [sum(1 << Ingred2ID[ingred] for ingred in ingred_set) for ingred_set in ingredient_sets_all])
    pool = []
    all_policy_indices = np.arange(len(ingredient_sets_all))
    np.random.shuffle(all_policy_indices)
    for i in range(pool_size):
        if i >= len(all_policy_indices):
            i = i % len(all_policy_indices)
        ingredient_support_set = ingredient_sets_all[all_policy_indices[i] % len(ingredient_sets_all)]
        print(f'Policy generated with support set {ingredient_support_set} and convention {None}')
        if gen_desire:
            policy = [int(ing in ingredient_support_set) for ing in ingredients]
        else:
            # policy = RuleBasedPolicy('minimum', np.random.rand() * p_max, np.random.rand() * p_max, 0.05, None,
            #                          env_name, ingredient_support_set=ingredient_support_set)
            # 2026-07-30: honor p_max (was hard-coded 0,0,0 — --p silently ignored on this
            # path). Mirrors generate_policy_pool_6recipe. Pass --p 0.0 for the old behavior.
            policy = RuleBasedPolicy('full', p_max, p_max, p_max, None,
                                     env_name, ingredient_support_set=ingredient_support_set)
        pool.append(policy)

    np.random.set_state(old_state)
    return pool


def get_train_eval_pool_4ing(args):
    """
    Create training and evaluation pools for 4-ingredient recipes.
    
    Recipes:
    R2: Lettuce + Tomato + Onion + Carrot
    R3: Lettuce + Potato + Onion + Broccoli
    R4: Tomato + Potato + Carrot + Broccoli
    """
    assert args.env_name == 'Overcooked'
    
    # Get the environment map from config
    with open(args.env_config, 'r') as env_config_file:
        env_map = yaml.safe_load(env_config_file)['mode']
        print('Using map', env_map, 'for 4-ingredient recipes')
    
    # Define the 3 ingredient sets for our 4-ingredient recipes
    ingredient_sets_4ing = [
        ['Lettuce', 'Tomato', 'Onion', 'Carrot'],       # R2
        ['Lettuce', 'Potato', 'Onion', 'Broccoli'],     # R3
        ['Tomato', 'Potato', 'Carrot', 'Broccoli'],     # R4
    ]
    
    # Convert to unique ingredient sets for policy support
    # For rule-based agents, we need unique ingredients per recipe
    ingredient_support_sets = [
        list(set(recipe)) for recipe in ingredient_sets_4ing
    ]
    
    policy_pool_train = []
    policy_pool_eval = []
    
    # Create rule-based opponents for each recipe
    # Both training and evaluation use the same 4 recipes
    p_max = args.p
    
    # Training pool: create one agent per recipe
    for i in range(min(args.train_pool_size, len(ingredient_support_sets))):
        ingredient_support_set = ingredient_support_sets[i]
        print(f'Creating training policy {i+1} with support set {ingredient_support_set}')
        
        policy = RuleBasedPolicy(
            policy_type,  # policy_type
            0,       # nav_p (no randomness in navigation)
            0,       # tar_p (no randomness in target selection)
            p_max,   # rand_p (minimal randomness)
            None,    # convention
            env_map,  # Use the environment map, not env_name
            ingredient_support_set=ingredient_support_set
        )
        policy_pool_train.append(policy)
    
    # Evaluation pool: create one agent per recipe
    for i in range(min(args.eval_pool_size, len(ingredient_support_sets))):
        ingredient_support_set = ingredient_support_sets[i]
        print(f'Creating evaluation policy {i+1} with support set {ingredient_support_set}')
        
        policy = RuleBasedPolicy(
            'full',  # policy_type
            0,       # nav_p
            0,       # tar_p
            p_max,   # rand_p
            None,    # convention
            env_map,  # Use the environment map, not env_name
            ingredient_support_set=ingredient_support_set
        )
        policy_pool_eval.append(policy)
    
    print(f'Created {len(policy_pool_train)} 4-ingredient training opponents with support sets:',
          [p.ingredient_support_set for p in policy_pool_train])
    print(f'Created {len(policy_pool_eval)} 4-ingredient evaluation opponents with support sets:',
          [p.ingredient_support_set for p in policy_pool_eval])

    return policy_pool_train, policy_pool_eval


# ======================================================================
# 6-recipe disjoint-capability domain (Exp 2)
# ======================================================================
# Pair structure (mirrors EXP2_ENV_SPEC.md):
#   R1 = LO (Lettuce, Onion)        R4 = CP (Cucumber, Pepper)
#   R2 = PB (Potato, Broccoli)      R5 = SM (Spinach, Mushroom)
#   R3 = TC (Tomato, Carrot)        R6 = PG (Pumpkin, Garlic)
#
# Disjoint L1 capability types:
#   R1R2: knows {LO, PB}            R3R4: knows {TC, CP}        R5R6: knows {SM, PG}
# Each type's ingredient set is disjoint from the others'.

from gym_cooking.environment.cooking_zoo import Ingred2ID_6recipe

INGREDIENT_PAIRS_6RECIPE = {
    'R1': ('Lettuce',  'Onion'),
    'R2': ('Potato',   'Broccoli'),
    'R3': ('Tomato',   'Carrot'),
    'R4': ('Cucumber', 'Pepper'),
    'R5': ('Spinach',  'Mushroom'),
    'R6': ('Pumpkin',  'Garlic'),
}

# L1 disjoint capability types -> the L0 pair types they train against
L1_DISJOINT_TYPE_TO_PAIRS = {
    'R1R2': ['R1', 'R2'],
    'R3R4': ['R3', 'R4'],
    'R5R6': ['R5', 'R6'],
    'ALL':  ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
    # single-recipe keys (2026-10-21): one L0 only, for the L2 FI curriculum phase 1 'single' variant
    'R1': ['R1'], 'R2': ['R2'], 'R3': ['R3'], 'R4': ['R4'], 'R5': ['R5'], 'R6': ['R6'],
}

# 6-recipe env obs shape (computed by the env; constant for the canonical map).
OBS_SHAPE_6RECIPE = (115,)


def generate_policy_pool_6recipe(pair_keys, p_max, env_name, policy_type='full'):
    """
    Build a list of RuleBasedPolicy L0 agents for the 6-recipe disjoint domain.

    Args:
        pair_keys: list of pair keys e.g. ['R1', 'R2'] or ['R1','R2','R3','R4','R5','R6'].
        p_max: stochasticity for nav/tar/rand (passed through; usually 0.0 for deterministic L0).
        env_name: map name (e.g. 'fc_small_divider_6recipe') so RuleBasedPolicy can derive divider geometry.

    Returns: list of RuleBasedPolicy, one per pair in pair_keys (each L0 knows exactly one recipe).
    """
    pool = []
    for key in pair_keys:
        if key not in INGREDIENT_PAIRS_6RECIPE:
            raise ValueError(f"Unknown 6-recipe pair key {key!r}; expected one of {list(INGREDIENT_PAIRS_6RECIPE)}")
        left, right = INGREDIENT_PAIRS_6RECIPE[key]
        print(f'6-recipe L0 policy with support set [{left}, {right}] ({key})')
        policy = RuleBasedPolicy(
            'full',
            # nav_p=0, tar_p=0, rand_p=p_max — matches the PC specialized pool.
            # Feeding p into all three (the old p_max,p_max,p_max) made the
            # disjoint-domain L0 wander so much it never delivered its half,
            # yielding 0.00 L1 success. rand_p alone (step-level noise, never an
            # interact action) is the only knob that adds real unpredictability
            # while staying recipe-safe; L1_PC_p05 trained to ~0.9 with it.
            0, 0, p_max,
            None,
            env_name,
            ingredient_support_set=[left, right],
            ingred_2_id=Ingred2ID_6recipe,
        )
        pool.append(policy)
    return pool


# ---------------------------------------------------------------------------------------------
# 6recipe_x4 domain (2026-10-28): six 4-ingredient recipes (2 LEFT + 2 RIGHT ingredients each) on
# the octahedron K_{2,2,2}: every ingredient belongs to exactly TWO recipes of DIFFERENT types, so one
# observed touch never identifies the partner's type; two touches from one recipe do. Types = antipodal
# pairs of recipes (disjoint ingredient sets), analogous to R1R2/R3R4/R5R6 in the 6recipe domain.
# Same 12 ingredients, same map (fc_small_divider_6recipe), same observation layout.
# ---------------------------------------------------------------------------------------------
X4_RECIPE_SIDES = {   # name: (left ingredients, right ingredients)  -- mirrors recipe_drawer.X4_RECIPES
    'X4_A1': (('Lettuce', 'Tomato'),   ('Onion', 'Carrot')),
    'X4_A2': (('Potato', 'Cucumber'),  ('Broccoli', 'Pepper')),
    'X4_B1': (('Lettuce', 'Spinach'),  ('Broccoli', 'Mushroom')),
    'X4_B2': (('Potato', 'Pumpkin'),   ('Onion', 'Garlic')),
    'X4_C1': (('Tomato', 'Spinach'),   ('Pepper', 'Garlic')),
    'X4_C2': (('Cucumber', 'Pumpkin'), ('Carrot', 'Mushroom')),
}
X4_TYPE_TO_RECIPES = {
    'A': ['X4_A1', 'X4_A2'],
    'B': ['X4_B1', 'X4_B2'],
    'C': ['X4_C1', 'X4_C2'],
    'ALL': ['X4_A1', 'X4_A2', 'X4_B1', 'X4_B2', 'X4_C1', 'X4_C2'],
}


def generate_policy_pool_x4(recipe_keys, p_max, env_name, policy_type='full'):
    """Rule-based L0s for the 6recipe_x4 domain: one per recipe, support set = its 4 ingredients.
    Each L0 can only reach its own side, so it performs the 2 ingredients on that side (fetch, chop,
    plate x2) and, on the RIGHT, delivers once the plate holds all four. Same noise convention as the
    6recipe pool (nav_p=0, tar_p=0, rand_p=p_max)."""
    pool = []
    for key in recipe_keys:
        if key not in X4_RECIPE_SIDES:
            raise ValueError(f"Unknown 6recipe_x4 recipe {key!r}; expected one of {list(X4_RECIPE_SIDES)}")
        left, right = X4_RECIPE_SIDES[key]
        print(f'6recipe_x4 L0 policy with support set {list(left + right)} ({key})')
        pool.append(RuleBasedPolicy('full', 0, 0, p_max, None, env_name,
                                    ingredient_support_set=list(left + right),
                                    ingred_2_id=Ingred2ID_6recipe,
                                    plate_take_requires_deliver=True))
    return pool


def get_train_eval_l0_pool_x4(args):
    """Train/eval L0 pools for 6recipe_x4 L1 training, driven by args.l0_pair_type:
    'A' | 'B' | 'C' -> the type's 2 L0s; 'ALL' -> all 6; a single recipe name -> 1 L0."""
    key = getattr(args, 'l0_pair_type', None)
    if key in X4_TYPE_TO_RECIPES:
        keys = X4_TYPE_TO_RECIPES[key]
    elif key in X4_RECIPE_SIDES:
        keys = [key]
    else:
        raise ValueError(f"--l0-pair-type must be one of {list(X4_TYPE_TO_RECIPES)} or {list(X4_RECIPE_SIDES)} "
                         f"for recipe_domain 6recipe_x4; got {key}")
    env_map = 'fc_small_divider_6recipe'
    pt = getattr(args, 'l0_policy_type', None) or 'full'
    train_pool = generate_policy_pool_x4(keys, p_max=args.p, env_name=env_map, policy_type=pt)
    eval_pool = generate_policy_pool_x4(keys, p_max=args.p, env_name=env_map, policy_type=pt)
    return train_pool, eval_pool


class TruncatedPolicy:
    """Wrap a policy so it acts for only the first `n_steps` of each episode, then NOOPs.

    Purpose: measure how much partner action a trained L1 needs before it reacts. With
    n_steps small the partner has barely revealed anything (an L0's first pickup typically
    lands around step 3-4), so this sweeps "probe strength" from a single action up to a
    full committed reveal. If L1 only succeeds at large n_steps, it needs its partner to
    COMMIT before responding -- which means probing cannot give L2 cheap information.
    """

    def __init__(self, inner, n_steps):
        self.inner = inner
        self.n_steps = int(n_steps)
        self.t = 0
        # pass through attributes the eval harness may read
        self.ingredient_support_set = getattr(inner, 'ingredient_support_set', None)
        self.ingredient_support_set_id = getattr(inner, 'ingredient_support_set_id', None)

    @property
    def _active(self):
        # overcooked_single.get_oppo_action() unwraps `_active` to decide whether to pass the
        # world (RuleBasedPolicy) or an obs array. Without this the wrapper is handed an obs.
        return self.inner

    def set_id(self, aid):
        self.inner.set_id(aid)

    def reset(self):
        self.t = 0
        if hasattr(self.inner, 'reset'):
            self.inner.reset()

    def __call__(self, world):
        self.t += 1
        if self.t <= self.n_steps:
            return self.inner(world)
        return 0   # NOOP: partner goes silent


class RecipeProber:
    """Cycles through revealing each of the 6 FI recipes, in randomized order.

    Each inner policy is a RuleBasedPolicy(policy_type='minimum') for ONE recipe: it fetches
    that recipe's ingredient to the divider but never plates it -- a reveal WITHOUT
    commitment. Every `dwell` steps the prober switches to the next recipe in its order.

    This is what a systematically-probing L2 would do. The measurement: does the trained L1
    engage only when one of ITS OWN two recipes is being revealed, and ignore the other four?
      non-zero success  -> L1 reacts to probes; probing can give L2 real information.
      zero success      -> L1 needs a committed partner; probing yields nothing.
    """

    def __init__(self, inner_policies, recipe_keys, dwell=10, order_seed=0):
        assert len(inner_policies) == len(recipe_keys)
        self.pols = inner_policies
        self.keys = list(recipe_keys)
        self.dwell = max(1, int(dwell))
        self._rng = np.random.RandomState(order_seed)
        self.order = self._rng.permutation(len(self.pols))
        self.t = 0
        self.revealed = []          # (step, recipe_key) trace, for analysis
        self.ingredient_support_set = getattr(inner_policies[0], 'ingredient_support_set', None)
        self.ingredient_support_set_id = getattr(inner_policies[0], 'ingredient_support_set_id', None)

    def _active_idx(self):
        return int(self.order[(self.t // self.dwell) % len(self.order)])

    @property
    def _active(self):
        # See TruncatedPolicy._active: makes the harness pass the world object.
        return self.pols[self._active_idx()]

    def set_id(self, aid):
        for p in self.pols:
            p.set_id(aid)

    def reset(self):
        self.t = 0
        self.order = self._rng.permutation(len(self.pols))
        self.revealed = []
        for p in self.pols:
            if hasattr(p, 'reset'):
                p.reset()

    def __call__(self, world):
        i = self._active_idx()
        if not self.revealed or self.revealed[-1][1] != self.keys[i]:
            self.revealed.append((self.t, self.keys[i]))
        self.t += 1
        return self.pols[i](world)


PREFIX_TIERS = ('pickup', 'show', 'chop', 'full')


def _prefix_support_set(ingredient, tier):
    """Support set for a diagnostic PREFIX of an L0 trajectory, for exactly one ingredient.

    The tiers are nested signals of increasing strength / decreasing reversibility:
      'pickup' -> Take(X) only. Probe walks to X, picks it up, holds it. Nothing is placed,
                  nothing chopped. Fully reversible.
      'show'   -> Take(X) + Put(X, Divider). X is parked on the shared divider, visible to the
                  partner from its own side. Still fully reversible (either agent can take it back).
      'chop'   -> Take(X) + Put(X, CutBoard) + Put(X, Divider). NOTE: RuleBasedPolicy.__call__
                  hard-chains Put(.,CutBoard) -> Chop -> Take regardless of the support set, so
                  this tier necessarily chops; omitting event('Chop', X) would NOT prevent it.
                  Chopping is irreversible, so this is a stronger signal but no longer a
                  reversible probe.

    Put(X, 'Plate') and Put('Plate', 'DeliverSquare') are absent from EVERY tier, so no tier can
    plate or deliver. No delivery is expected during a prefix -- the measurement is the partner's
    reaction, not task success.
    """
    if tier == 'pickup':
        return [event('Take', ingredient)]
    if tier == 'show':
        return [event('Take', ingredient), event('Put', ingredient, 'Divider')]
    if tier == 'chop':
        return [event('Take', ingredient),
                event('Put', ingredient, 'CutBoard'),
                event('Put', ingredient, 'Divider')]
    raise ValueError(f"tier must be one of {PREFIX_TIERS[:3]} (or 'full' for the control); got {tier!r}")


class PrefixProbePolicy:
    """A scripted leader that executes only a PREFIX of one recipe, then HOLDS the signal.

    Wraps a RuleBasedPolicy(policy_type='specified') whose support set is one tier from
    _prefix_support_set. Once the prefix has physically completed, this policy returns NOOP for
    the rest of the episode. The freeze is required, not cosmetic: a 'specified' policy has no
    guard against re-taking an ingredient off the divider (that guard only exists for
    'medium'/'minimum'), so without it the probe oscillates Take/Put and the signal flickers
    instead of standing still for the partner to read.

    Measurement: every step this policy writes the EGO partner's state (held ingredient + chopped
    flag, and the divider contents) to a JSONL trace. The probe sees the `world` object each call,
    so this works even when the env lives in a SubprocVecEnv worker -- each worker appends to its
    own pid-suffixed file and the analyzer merges them.

    What the trace answers: does the frozen L1 pick up / chop the MATCHING ingredient of the
    proposed recipe, and does it do so only for recipes inside its capability set? Delivery
    success is deliberately not the metric.
    """

    def __init__(self, inner, ingredient, tier, recipe_key, trace_path=None, tag=''):
        self.inner = inner
        self.ingredient = ingredient
        self.tier = tier
        self.recipe_key = recipe_key
        self.trace_path = trace_path
        self.tag = tag
        self.t = 0
        self.episode = 0
        self.frozen_at = None
        self._fh = None
        self.ingredient_support_set = getattr(inner, 'ingredient_support_set', None)
        self.ingredient_support_set_id = getattr(inner, 'ingredient_support_set_id', None)

    @property
    def _active(self):
        # overcooked_single.get_oppo_action() unwraps `_active` to decide whether to hand over the
        # world object or an obs array. Without this the wrapper is handed an obs array.
        return self.inner

    def set_id(self, aid):
        self.inner.set_id(aid)

    # ---- prefix-completion test, read off the world ----------------------------------------
    def _divider_items(self, world):
        out = []
        for loc in self.inner.divider_loc:
            for obj in world.get_objects_at(loc, DynamicObject):
                nm = type(obj).__name__
                done = bool(obj.done()) if hasattr(obj, 'done') else False
                out.append((nm, done))
        return out

    def _prefix_complete(self, world):
        me = world.agents[self.inner.agent_id]
        holding_target = (me.holding is not None
                          and isinstance(me.holding, StringToClass[self.ingredient]))
        if self.tier == 'pickup':
            return holding_target
        items = self._divider_items(world)
        if self.tier == 'show':
            return (not holding_target) and any(n == self.ingredient for n, _ in items)
        if self.tier == 'chop':
            return (not holding_target) and any(n == self.ingredient and d for n, d in items)
        return False

    # ---- trace -----------------------------------------------------------------------------
    def _record(self, world):
        ego_id = 1 - self.inner.agent_id
        ego = world.agents[ego_id]
        h = ego.holding
        row = {
            'tag': self.tag, 'recipe': self.recipe_key, 'tier': self.tier,
            'probe_ingredient': self.ingredient,
            'ep': self.episode, 't': self.t,
            'probe_frozen': self.frozen_at is not None,
            'ego_holding': (type(h).__name__ if h is not None else None),
            'ego_holding_chopped': (bool(h.done()) if (h is not None and hasattr(h, 'done')) else False),
            'divider': self._divider_items(world),
        }
        self._write(row)

    def _flush(self):
        if self._fh is not None:
            self._fh.flush()

    def _write(self, row):
        # Written immediately rather than buffered per episode: reset() fires at the START of an
        # episode, so a buffer would silently drop the LAST episode of every run.
        if not self.trace_path:
            return
        if self._fh is None:
            import os
            d = os.path.dirname(self.trace_path)
            if d:
                os.makedirs(d, exist_ok=True)
            self._fh = open(self.trace_path, 'a')
        import json
        self._fh.write(json.dumps(row) + '\n')

    def reset(self):
        # overcooked_single.reset() calls this once per EPISODE (not per meta-episode), exactly as
        # it does for a real L0 partner, so the ego's 5-episode history protocol is untouched.
        self._flush()
        self.episode += 1
        self.t = 0
        self.frozen_at = None
        self.inner.cur_event = None
        self.inner.refresh_event()
        if hasattr(self.inner, 'reset'):
            self.inner.reset()

    def __call__(self, world):
        self._record(world)
        self.t += 1
        if self.frozen_at is not None:
            return 0
        if self._prefix_complete(world):
            self.frozen_at = self.t
            return 0
        return self.inner(world)


def make_prefix_probe(recipe_key, tier, p_max, env_name, side, trace_path=None, tag=''):
    """Build the partner for one (recipe, tier) diagnostic cell.

    side: 0 if the probe is the LEFT agent (ego L1 is RIGHT / player-id 1), else 1. It selects
    which half of the recipe pair the probe carries -- the partner is expected to supply the other.

    tier 'full' returns an ordinary complete L0 for that single recipe: the POSITIVE CONTROL,
    the one cell where delivery should happen.
    """
    left, right = INGREDIENT_PAIRS_6RECIPE[recipe_key]
    mine, theirs = (left, right) if side == 0 else (right, left)
    if tier == 'full':
        pol = RuleBasedPolicy('full', 0, 0, p_max, None, env_name,
                              ingredient_support_set=[left, right],
                              ingred_2_id=Ingred2ID_6recipe)
        print(f'[probe] {recipe_key} tier=full (positive control) mine={mine} theirs={theirs}')
        return pol
    ss = _prefix_support_set(mine, tier)
    inner = RuleBasedPolicy('specified', 0, 0, p_max, None, env_name,
                            support_set=ss, event_probs=[1.0 / len(ss)] * len(ss),
                            ingredient_support_set=[left, right],
                            ingred_2_id=Ingred2ID_6recipe)
    print(f'[probe] {recipe_key} tier={tier} mine={mine} theirs={theirs} '
          f'events={[str(e) for e in ss]}')
    return PrefixProbePolicy(inner, mine, tier, recipe_key, trace_path=trace_path, tag=tag)


def get_train_eval_l0_pool_6recipe(args):
    """
    Build train/eval L0 pools for 6-recipe L1 training. Driven by args.l0_pair_type:
      'R1R2' -> [L0(R1), L0(R2)]
      'R3R4' -> [L0(R3), L0(R4)]
      'R5R6' -> [L0(R5), L0(R6)]
      'ALL'  -> all 6
    Train pool and eval pool are identical (rule-based, deterministic).
    """
    pair_type = getattr(args, 'l0_pair_type', None)
    if pair_type is None or pair_type not in L1_DISJOINT_TYPE_TO_PAIRS:
        raise ValueError(f"--l0-pair-type must be one of {list(L1_DISJOINT_TYPE_TO_PAIRS)}; got {pair_type}")
    pair_keys = L1_DISJOINT_TYPE_TO_PAIRS[pair_type]
    env_map = 'fc_small_divider_6recipe'
    pt = getattr(args, 'l0_policy_type', None) or 'full'
    train_pool = generate_policy_pool_6recipe(pair_keys, p_max=args.p, env_name=env_map, policy_type=pt)
    eval_pool  = generate_policy_pool_6recipe(pair_keys, p_max=args.p, env_name=env_map, policy_type=pt)
    if getattr(args, 'l0_prober', False):
        # Probe over ALL SIX recipes, not just this type's pair: the point is to see which
        # reveals the L1 engages with.
        all_keys = ['R1','R2','R3','R4','R5','R6']
        dwell = getattr(args, 'l0_prober_dwell', 10) or 10
        def _mk():
            pols = generate_policy_pool_6recipe(all_keys, p_max=args.p, env_name=env_map,
                                                policy_type='minimum')
            return RecipeProber(pols, all_keys, dwell=dwell, order_seed=args.seed or 0)
        print(f'L0 replaced by RecipeProber over all 6 recipes, dwell={dwell} steps')
        return [_mk()], [_mk()]
    prefix_tier = getattr(args, 'l0_prefix_tier', None)
    if prefix_tier:
        # Focused partner-responsiveness diagnostic: ONE recipe, ONE prefix tier, so the
        # per-opponent array is unnecessary and the trace is unambiguous.
        rk = getattr(args, 'l0_prefix_recipe', None)
        if rk not in INGREDIENT_PAIRS_6RECIPE:
            raise ValueError(f'--l0-prefix-recipe must be one of {list(INGREDIENT_PAIRS_6RECIPE)}; got {rk}')
        side = 1 - int(getattr(args, 'player_id', 1))   # probe takes the seat the ego is not in
        tp = getattr(args, 'probe_trace_path', None)
        if tp:
            import os
            tp = f'{tp}.pid{os.getpid()}.jsonl'
        tag = getattr(args, 'probe_tag', '') or ''
        def _mkp():
            return make_prefix_probe(rk, prefix_tier, args.p, env_map, side,
                                     trace_path=tp, tag=tag)
        print(f'L0 replaced by PrefixProbePolicy recipe={rk} tier={prefix_tier} side={side} '
              f'(declared type {pair_type}; supported={pair_type != "ALL" and rk in L1_DISJOINT_TYPE_TO_PAIRS[pair_type]})')
        return [_mkp()], [_mkp()]

    trunc = getattr(args, 'l0_truncate_steps', None)
    if trunc:
        train_pool = [TruncatedPolicy(p, trunc) for p in train_pool]
        eval_pool  = [TruncatedPolicy(p, trunc) for p in eval_pool]
        print(f'L0 partners TRUNCATED to the first {trunc} steps, then NOOP')
    return train_pool, eval_pool


def get_train_eval_l1_disjoint_pool(args, train_spec, eval_spec):
    """
    Load disjoint-capability L1 partners (3 types: R1R2 / R3R4 / R5R6) for L2 training
    in the 6-recipe domain.

    Args:
        train_spec: "R1R2:1,2,3,4,5,6,7;R3R4:1,2,3,4,5,6,7;R5R6:1,2,3,4,5,6,7"
        eval_spec:  "R1R2:8,9,10;R3R4:8,9,10;R5R6:8,9,10"

    Returns: (train_pool, eval_pool), each a list of PretrainedPolicy_test.
    Loads checkpoints from logs/Overcooked/ppo_L1_disjoint/{TYPE}/seed{N}_seed{N}/ppo/latest.pt
    Passes obs_shape=(115,) to PretrainedPolicy_test for the 6-recipe history buffer.
    """
    # --l1-disjoint-base-dir overrides the default so L2 can train against a specific
    # FI pool (e.g. june_new_exp/L1_FI_pace_p0/right). Default preserves old behavior.
    override_base = getattr(args, 'l1_disjoint_base_dir', None)
    base_location = override_base or \
        f"{PACE_ROOT}/logs/Overcooked/ppo_L1_disjoint"
    print('=' * 70)
    print('Loading L1 DISJOINT pool for 6-recipe L2 training')
    print('=' * 70)
    print(f'Base location: {base_location}' + ('  (overridden)' if override_base else '  (default)'))
    # --feed-partner-history: partners get their history written during L2 training (see
    # PretrainedPolicy_test.feed_history). Their buffer must then hold a full episode, so take
    # the horizon from the env config (FI H=80). Off by default = old behaviour.
    feed_history = bool(getattr(args, 'feed_partner_history', False))
    feed_kwargs = {}
    if feed_history:
        import yaml as _yaml
        with open(args.env_config) as _f:
            _horizon = _yaml.safe_load(_f).get('horizon')
        feed_kwargs = {'feed_history': True, 'max_episode_length': int(_horizon)}
        print(f'Partner history: FED during training (horizon {_horizon})')
    else:
        print('Partner history: not written (default)')

    # Resolve a checkpoint under {base}/{TYPE}/, trying the consolidated seed{N}/ layout
    # first then the raw seed{N}_seed{N}/ layout, and preferring the explicit 30M ckpt
    # over latest.pt so the pool is reproducible.
    def _resolve_ckpt(agent_type, seed):
        return _resolve_l1_ckpt(f"{base_location}/{agent_type}", seed, '30000000.pt')

    def _load_spec(label, spec):
        pool = []
        print(f'\n{label} pool:')
        for type_spec in spec.split(';'):
            agent_type, seeds_str = type_spec.strip().split(':')
            seeds = [int(s.strip()) for s in seeds_str.split(',')]
            if agent_type not in L1_DISJOINT_TYPE_TO_PAIRS or agent_type == 'ALL':
                raise ValueError(f"L1 disjoint type {agent_type!r} not allowed (must be R1R2/R3R4/R5R6)")
            for seed in seeds:
                ckpt = _resolve_ckpt(agent_type, seed)
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(f"L1 disjoint checkpoint not found: {ckpt}")
                policy = PretrainedPolicy_test(
                    latent_training=args.latent_training,
                    history_size=args.history_size,
                    has_rew_done=args.has_rew_done,
                    has_meta_time_step=args.has_meta_time_step,
                    include_current_episode=args.include_current_episode,
                    merge_encoder_computation=args.merge_encoder_computation,
                    last_episode_only=args.last_episode_only,
                    pop_oldest_episode=args.pop_oldest_episode,
                    self_obs_mode=args.self_obs_mode,
                    model_path=ckpt,
                    eval_history=True,
                    agent_id=1 - args.player_id,
                    obs_shape=OBS_SHAPE_6RECIPE,
                    **feed_kwargs,
                )
                pool.append(policy)
                print(f'  ✓ {agent_type} seed {seed}  <- {os.path.relpath(ckpt, base_location)}')
        return pool

    train_pool = _load_spec('Train', train_spec)
    eval_pool  = _load_spec('Eval',  eval_spec)
    return train_pool, eval_pool
