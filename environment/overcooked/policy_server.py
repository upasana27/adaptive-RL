import os.path
import random
import re
import numpy as np
import torch
import yaml
from .gym_cooking.cooking_world.world_objects import *
from .gym_cooking.cooking_world.abstract_classes import *
from .gym_cooking.environment.cooking_zoo import Ingred2ID
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
        self.agent_id = agent_id
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

    def __call__(self, observation, previous_step):

        with torch.no_grad():
            if self.eval_history is not None:
                all_agent_indices = torch.arange(1)
                indices = self.history.get_all_current_indices()
                history = (self.history, (all_agent_indices,) + indices)
                #history = None
                # print(observation.shape)
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
        return action.squeeze(-1).item()
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
                complete_reward=10.0, step_cost=0.05,model_path=None,eval_history=True,agent_id=0, device='cuda'):
        # Lazy init. The model is loaded only when the first observation is received, in the environment process
        # This guarantees that no tensor needs to be moved across processes
        self.model_path = model_path
        self.agent_id = agent_id
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
        self.self_obs_mode = self_obs_mode
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
                obs_shape=(73,),
                act_shape=tuple(),
                max_episode_length=40,
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

    def __call__(self, observation, previous_step):

        with torch.no_grad():
            if self.eval_history is not None:
                all_agent_indices = torch.arange(1)
                indices = self.history.get_all_current_indices()
                history = (self.history, (all_agent_indices,) + indices)
                #history = None
                # print(observation.shape)
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
        return action.squeeze(-1).item()
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
    def reset(self):
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
                obs_shape=(73,),
                act_shape=tuple(),
                max_episode_length=40,
                merge_encoder_computation=self.merge_encoder_computation,
                last_episode_only=self.last_episode_only,
                pop_oldest_episode=self.pop_oldest_episode,
            )
            self.history.to('cuda')
        else:
            self.history = None

        
class RuleBasedPolicy:
    def __init__(self, policy_type, nav_p, tar_p, rand_p, convention, env_name, support_set=None,
                 ingredient_support_set=None, event_probs=None):
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
        ingred_id_set = [Ingred2ID[ingred] for ingred in self.ingredient_support_set]
        self.ingredient_support_set_id = tuple(int(i in ingred_id_set) for i in range(len(Ingred2ID)))
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
                        if len(in_plate) == 0 or (len(in_plate) == 1 and len(self.ingredient_support_set) > 1):
                            # recipe not satisfied
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
                raise NotImplementedError
        else:
            # stop and wait
            action = 0
        return action

    def get_action_from_event(self, world):
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
                else:
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
                deliver_loc = world.world_objects['DeliverSquare'][0].location
                action = self.get_navigation_action(world, self_loc, deliver_loc)
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
            # elif self.cur_event.action == "Take" and self.cur_event.dynamic_obj != "Plate":
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
from pathlib import Path
def get_train_eval_cross_seed_pool(args):
    # pick all the latest agents from logs
    train_pool = []
    eval_pool = []
    location = "/home/asurite.ad.asu.edu/ubiswas2/PACE/logs/Overcooked/cross_seed_eval/"
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

def get_train_eval_L1seed7_pool(args):
    """Load L1 seed7 opponents (unseen) for evaluation."""
    train_pool = []
    eval_pool = []
    location = "/home/asurite.ad.asu.edu/ubiswas2/PACE/logs/Overcooked/"
    latent_training = True
    history_size = 5
    has_rew_done = False
    has_meta_time_step = False
    include_current_episode = True
    last_episode_only = False
    merge_encoder_computation = True
    pop_oldest_episode = True
    self_obs_mode = True
    
    # List of specific L1 seed7 opponent directories
    opponent_dirs = [
        "ppo_pace_all3_left_seed7",
        "ppo_pace_excl_LettuceOnion_left_seed7",
        "ppo_pace_excl_PotatoBroccoli_left_seed7",
        "ppo_pace_excl_TomatoCarrot_left_seed7",
        "ppo_pace_all3_right_seed7",
        "ppo_pace_excl_LettuceOnion_right_seed7",
        "ppo_pace_excl_PotatoBroccoli_right_seed7",
        "ppo_pace_excl_TomatoCarrot_right_seed7",
    ]
    
    for opp_dir in opponent_dirs:
        model_path = os.path.join(location, opp_dir, "ppo", "latest.pt")
        if os.path.exists(model_path):
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
                model_path=model_path,
                eval_history=False
            )
            eval_pool.append(policy)
    
    print(f"Loaded {len(eval_pool)} L1 seed7 opponent policies")
    return train_pool, eval_pool

def get_train_eval_L1seed6_pool(args):
    """Load L1 seed6 opponents (training partners) for evaluation."""
    train_pool = []
    eval_pool = []
    location = "/home/asurite.ad.asu.edu/ubiswas2/PACE/logs/Overcooked/level_1/"
    latent_training = True
    history_size = 5
    has_rew_done = False
    has_meta_time_step = False
    include_current_episode = True
    last_episode_only = False
    merge_encoder_computation = True
    pop_oldest_episode = True
    self_obs_mode = True
    
    # List of L1 seed6 opponent directories (4 left + 4 right)
    opponent_dirs = [
        "ppo_pace_all3_left_seed6",
        "ppo_pace_excl_LettuceOnion_left_seed6",
        "ppo_pace_excl_PotatoBroccoli_left_seed6",
        "ppo_pace_excl_TomatoCarrot_left_seed6",
        "ppo_pace_all3_right_seed6",
        "ppo_pace_excl_LettuceOnion_right_seed6",
        "ppo_pace_excl_PotatoBroccoli_right_seed6",
        "ppo_pace_excl_TomatoCarrot_right_seed6",
    ]
    
    for opp_dir in opponent_dirs:
        model_path = os.path.join(location, opp_dir, "ppo", "latest.pt")
        if os.path.exists(model_path):
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
                model_path=model_path,
                eval_history=False
            )
            eval_pool.append(policy)
    
    print(f"Loaded {len(eval_pool)} L1 seed6 opponent policies")
    return train_pool, eval_pool

def get_train_eval_L1_multiseed_pool(args, seeds=[1, 2, 3, 4, 5]):
    """
    Load L1 opponents from multiple seeds for robust L2 training.
    During training, L2 will randomly sample from this diverse pool,
    improving generalization to unseen partners.
    
    Args:
        args: Training arguments
        seeds: List of seeds to load agents from (default: [1,2,3,4,5])
    
    Returns:
        train_pool: Combined pool of L1 agents from all seeds
        eval_pool: Same as train_pool (for compatibility)
    """
    import random
    
    train_pool = []
    eval_pool = []
    
    # Configuration for loading L1 agents
    latent_training = True
    history_size = 5
    has_rew_done = False
    has_meta_time_step = False
    include_current_episode = True
    last_episode_only = False
    merge_encoder_computation = True
    pop_oldest_episode = True
    self_obs_mode = True
    
    # Agent configurations (recipe type, player_id)
    agent_configs = [
        # LEFT agents (player-id 0)
        ("L1_all3_left", "cross", 0),
        ("L1_excl_LettuceOnion_left", "excl_LettuceOnion", 0),
        ("L1_excl_PotatoBroccoli_left", "excl_PotatoBroccoli", 0),
        ("L1_excl_TomatoCarrot_left", "excl_TomatoCarrot", 0),
        # RIGHT agents (player-id 1)
        ("L1_all3_right", "cross", 1),
        ("L1_excl_LettuceOnion_right", "excl_LettuceOnion", 1),
        ("L1_excl_PotatoBroccoli_right", "excl_PotatoBroccoli", 1),
        ("L1_excl_TomatoCarrot_right", "excl_TomatoCarrot", 1),
    ]
    
    print(f"Loading L1 agents from {len(seeds)} seeds: {seeds}")
    
    # Load agents from each seed
    for seed in seeds:
        seed_agents_loaded = 0
        for exp_name, recipe_type, player_id in agent_configs:
            # Construct path to seed-specific agent in organized structure
            # Path: logs/Overcooked/L1_multiseed/seed1/L1_all3_left/ppo/30000000.pt
            model_path = f"/home/asurite.ad.asu.edu/ubiswas2/PACE/logs/Overcooked/L1_multiseed/seed{seed}/{exp_name}/ppo/30000000.pt"
            
            # Check if model exists
            if not os.path.exists(model_path):
                print(f"  Warning: {model_path} not found, skipping...")
                continue
            
            # Create policy
            policy = Policy(
                args,
                player_id=player_id,
                recipe_type=recipe_type,
                latent_training=latent_training,
                history_size=history_size,
                has_rew_done=has_rew_done,
                has_meta_time_step=has_meta_time_step,
                include_current_episode=include_current_episode,
                merge_encoder_computation=merge_encoder_computation,
                last_episode_only=last_episode_only,
                pop_oldest_episode=pop_oldest_episode,
                self_obs_mode=self_obs_mode,
                model_path=model_path,
                eval_history=False  # Use history during training
            )
            train_pool.append(policy)
            eval_pool.append(policy)
            seed_agents_loaded += 1
        
        print(f"  Seed {seed}: Loaded {seed_agents_loaded} agents")
    
    print(f"\nTotal loaded: {len(train_pool)} L1 agents from {len(seeds)} seeds")
    print(f"Pool diversity: {len(train_pool)} unique agent variants")
    
    # Shuffle the pool for random sampling during training
    random.shuffle(train_pool)
    random.shuffle(eval_pool)
    
    return train_pool, eval_pool

def get_train_eval_level_1_pool(args):
    # pick all the latest agents from logs
    train_pool = []
    eval_pool = []
    location = "/home/asurite.ad.asu.edu/ubiswas2/PACE/logs/Overcooked/level_1/"
    for root, dirs,files in os.walk(location):
        for file in files:
            if file == "latest.pt":
                file_path = os.path.join(root,file)
                policy = PretrainedPolicy(args,model_path=file_path, eval_history=False)
                eval_pool.append(policy)
    
    for root, dirs,files in os.walk(location):
        for file in files:           
            if file == "25008000.pt":
                file_path = os.path.join(root,file)
                policy = PretrainedPolicy(args,model_path=file_path, eval_history=False)
                eval_pool.append(policy)
    # eval_pool = eval_pool[0:2]
    return train_pool, eval_pool

def get_train_eval_pool_4ing(args):
    """
    Create training pool for 4-ingredient recipes.
    Each 4-ingredient recipe (LLOC, LTOC, LPOB, TPCB) is a single goal.
    Rule-based agents will be created for each recipe.
    """
    policy_pool_train = []
    policy_pool_eval = []
    
    with open(args.env_config, 'r') as env_config_file:
        env_config = yaml.safe_load(env_config_file)
        env_map = env_config['mode']
        recipe_names = env_config['recipes']
    
    print('=' * 80)
    print('L1 4-INGREDIENT TRAINING MODE')
    print(f'Using map: {env_map}')
    print(f'Recipe type: {args.recipe_type}')
    print(f'Recipes: {recipe_names}')
    print('=' * 80)
    
    # Define the ingredient sets for each 4-ingredient recipe
    # LLOC: Lettuce + Lettuce + Onion + Carrot
    # LTOC: Lettuce + Tomato + Onion + Carrot
    # LPOB: Lettuce + Potato + Onion + Broccoli
    # TPCB: Tomato + Potato + Carrot + Broccoli
    recipe_ingredient_map = {
        'LLOCSalad': ['Lettuce', 'Lettuce', 'Onion', 'Carrot'],
        'LTOCSalad': ['Lettuce', 'Tomato', 'Onion', 'Carrot'],
        'LPOBSalad': ['Lettuce', 'Potato', 'Onion', 'Broccoli'],
        'TPCBSalad': ['Tomato', 'Potato', 'Carrot', 'Broccoli']
    }
    
    ingredient_sets = [recipe_ingredient_map[recipe] for recipe in recipe_names]
    
    print(f'Ingredient sets: {ingredient_sets}')
    print(f'Creating {args.train_pool_size} rule-based training opponents')
    print(f'Creating {args.eval_pool_size} rule-based eval opponents')
    
    # Create training pool
    i = 0
    args.p = 0
    while i < args.train_pool_size:
        for recipe_ingredients in ingredient_sets:
            if i >= args.train_pool_size:
                break
            policy = RuleBasedPolicy('full', 0, 0, 0, None,
                                     env_map, ingredient_support_set=recipe_ingredients)
            policy_pool_train.append(policy)
            i += 1
    
    # Create eval pool
    j = 0
    while j < args.eval_pool_size:
        for recipe_ingredients in ingredient_sets:
            if j >= args.eval_pool_size:
                break
            policy = RuleBasedPolicy('full', 0, 0, 0, None,
                                     env_map, ingredient_support_set=recipe_ingredients)
            policy_pool_eval.append(policy)
            j += 1
    
    print(f'Put {len(policy_pool_train)} rule-based opponents into train pool, ingredient support sets:',
          [p.ingredient_support_set for p in policy_pool_train])
    print(f'Put {len(policy_pool_eval)} rule-based opponents into eval pool, ingredient support sets:',
          [p.ingredient_support_set for p in policy_pool_eval])
    print('=' * 80)
    
    return policy_pool_train, policy_pool_eval

def get_train_eval_pool(args):
    assert args.env_name == 'Overcooked'

    
    # L2 Multi-Seed Training: Load L1 agents from multiple seeds for better generalization
    if hasattr(args, 'use_multiseed_pool') and args.use_multiseed_pool:
        print('=' * 80)
        print('L2 MULTI-SEED TRAINING MODE')
        print('=' * 80)
        seeds = getattr(args, 'multiseed_list', [1, 2, 3, 4, 5])
        print(f'Loading L1 agents from seeds: {seeds}')
        print(f'This will improve generalization to unseen partners')
        print('=' * 80)
        
        train_pool, eval_pool = get_train_eval_L1_multiseed_pool(args, seeds=seeds)
        
        # Adjust pool sizes if needed
        if len(train_pool) < args.train_pool_size:
            print(f'Warning: Only {len(train_pool)} agents loaded, but train_pool_size={args.train_pool_size}')
            print(f'Adjusting train_pool_size to {len(train_pool)}')
            args.train_pool_size = len(train_pool)
        elif len(train_pool) > args.train_pool_size:
            print(f'Note: {len(train_pool)} agents available, using {args.train_pool_size} for training')
            # Randomly sample train_pool_size agents
            import random
            indices = random.sample(range(len(train_pool)), args.train_pool_size)
            train_pool = [train_pool[i] for i in indices]
        
        if args.eval_pool_size > 0:
            if len(eval_pool) < args.eval_pool_size:
                print(f'Warning: Only {len(eval_pool)} agents for eval, but eval_pool_size={args.eval_pool_size}')
                args.eval_pool_size = len(eval_pool)
            elif len(eval_pool) > args.eval_pool_size:
                import random
                indices = random.sample(range(len(eval_pool)), args.eval_pool_size)
                eval_pool = [eval_pool[i] for i in indices]
        
        print(f'Final pool sizes: train={len(train_pool)}, eval={len(eval_pool)}')
        return train_pool, eval_pool
    
    # L2 Training: Load L1 checkpoints as opponents
    if hasattr(args, 'l1_pool_path') and args.l1_pool_path is not None:
        print(f'L2 Training Mode: Loading L1 checkpoints from {args.l1_pool_path}')
        l1_paths = [path.strip() for path in args.l1_pool_path.split(',')]
        assert len(l1_paths) == args.train_pool_size, \
            f'Number of L1 checkpoints ({len(l1_paths)}) does not match train_pool_size ({args.train_pool_size})'
        
        policy_pool_train = []
        for path in l1_paths:
            # Load L1 agent using PretrainedPolicy_test (with history but eval_history=False means frozen)
            l1_policy = PretrainedPolicy_test(
                latent_training=args.latent_training,
                history_size=args.history_size,
                has_rew_done=args.has_rew_done if hasattr(args, 'has_rew_done') else False,
                has_meta_time_step=args.has_meta_time_step if hasattr(args, 'has_meta_time_step') else False,
                include_current_episode=args.include_current_episode,
                merge_encoder_computation=args.merge_encoder_computation,
                last_episode_only=args.last_episode_only if hasattr(args, 'last_episode_only') else False,
                pop_oldest_episode=args.pop_oldest_episode if hasattr(args, 'pop_oldest_episode') else False,
                self_obs_mode=args.self_obs_mode,
                model_path=path,
                eval_history=False  # Frozen L1 - has history buffer but doesn't update it
            )
            policy_pool_train.append(l1_policy)
            print(f'Loaded L1 checkpoint with history: {path}')
        
        # Add same L1 agents to eval pool if eval_pool_size > 0
        policy_pool_eval = []
        if args.eval_pool_size > 0:
            for path in l1_paths:
                l1_policy_eval = PretrainedPolicy_test(
                    latent_training=args.latent_training,
                    history_size=args.history_size,
                    has_rew_done=args.has_rew_done if hasattr(args, 'has_rew_done') else False,
                    has_meta_time_step=args.has_meta_time_step if hasattr(args, 'has_meta_time_step') else False,
                    include_current_episode=args.include_current_episode,
                    merge_encoder_computation=args.merge_encoder_computation,
                    last_episode_only=args.last_episode_only if hasattr(args, 'last_episode_only') else False,
                    pop_oldest_episode=args.pop_oldest_episode if hasattr(args, 'pop_oldest_episode') else False,
                    self_obs_mode=args.self_obs_mode,
                    model_path=path,
                    eval_history=False
                )
                policy_pool_eval.append(l1_policy_eval)
            print(f'Put {len(policy_pool_eval)} L1 agents into eval pool')
        
        print(f'Put {len(policy_pool_train)} L1 agents (with history) into train pool')
        return policy_pool_train, policy_pool_eval
    
    # L1 Training with 4-ingredient recipes: Special handling
    if hasattr(args, 'exp_name') and 'L1_4ing' in args.exp_name:
        return get_train_eval_pool_4ing(args)
    
    # L1 Training: Use rule-based opponents
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
    
    # Filter recipes based on experiment name (for ablation studies)
    # By default, use all three recipe combinations
    ingredient_set_train = ingredient_sets_all.copy()  # [Lettuce+Onion, Potato+Broccoli, Tomato+Carrot]
    ingredient_set_eval = ingredient_sets_all.copy()   # [Lettuce+Onion, Potato+Broccoli, Tomato+Carrot]
    
    # Check if we should exclude a specific recipe type
    if hasattr(args, 'exp_name'):
        if 'excl_TomatoCarrot' in args.exp_name:
            # Exclude Tomato+Carrot (index 2), keep Lettuce+Onion and Potato+Broccoli
            ingredient_set_train = [ingredient_sets_all[0], ingredient_sets_all[1]]
            ingredient_set_eval = [ingredient_sets_all[0], ingredient_sets_all[1]]
            print("EXCLUDING Tomato+Carrot - Training on: Lettuce+Onion, Potato+Broccoli")
        elif 'excl_PotatoBroccoli' in args.exp_name:
            # Exclude Potato+Broccoli (index 1), keep Lettuce+Onion and Tomato+Carrot
            ingredient_set_train = [ingredient_sets_all[0], ingredient_sets_all[2]]
            ingredient_set_eval = [ingredient_sets_all[0], ingredient_sets_all[2]]
            print("EXCLUDING Potato+Broccoli - Training on: Lettuce+Onion, Tomato+Carrot")
        elif 'excl_LettuceOnion' in args.exp_name:
            # Exclude Lettuce+Onion (index 0), keep Potato+Broccoli and Tomato+Carrot
            ingredient_set_train = [ingredient_sets_all[1], ingredient_sets_all[2]]
            ingredient_set_eval = [ingredient_sets_all[1], ingredient_sets_all[2]]
            print("EXCLUDING Lettuce+Onion - Training on: Potato+Broccoli, Tomato+Carrot")
    
    print(ingredient_set_train)
    print(ingredient_set_eval)
    i = 0
    j = 0
    args.p = 0
    while i < args.train_pool_size:
        for recipe in ingredient_set_train:
            policy = RuleBasedPolicy('full',0, 0, 0, None,
                                     env_map, ingredient_support_set=recipe)
            policy_pool_train.append(policy)
            i = i + 1
    # Use eval_pool_size from command-line arguments
    while j < args.eval_pool_size:
        for recipe in ingredient_set_eval:
            policy = RuleBasedPolicy('full',0, 0, 0, None,
                                     env_map, ingredient_support_set=recipe)
            policy_pool_eval.append(policy)
            j = j + 1
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
            policy = RuleBasedPolicy('full', 0, 0, 0, None,
                                     env_name, ingredient_support_set=ingredient_support_set)
        pool.append(policy)

    np.random.set_state(old_state)
    return pool