import os.path
import random
import re
import numpy as np
import torch
import yaml
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_world.abstract_classes import *
from gym_cooking.environment.cooking_zoo import Ingred2ID
from learning.model import LatentActor
import subprocess

static_objects = ['CutBoard','DeliverSquare','Divider','Plate']
ingredients = ['Lettuce','Tomato','Potato','Onion','Carrot','Broccoli']
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
    def __init__(self, model_path, agent_id, is_self_play=True, batch_size=1, device='cpu'):
        # Lazy init. The model is loaded only when the first observation is received, in the environment process
        # This guarantees that no tensor needs to be moved across processes
        self.model_path = model_path
        self.agent_id = agent_id
        self.actor = self.rnn_states = self.rnn_hidden_dim = None
        self.is_self_play = is_self_play
        self.batch_size = batch_size
        self.device = device

    def set_id(self, aid):
        self.agent_id = aid

    def __call__(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
        if self.batch_size == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action, _, self.rnn_states, _ = self.actor.act(obs, self.rnn_states,
                                                           torch.ones(self.batch_size, 1, device=self.device),
                                                           None, deterministic=False)
        return action.item() if self.batch_size == 1 else action.squeeze(-1)

    def reset(self):
        if self.actor is None:
            # One-time initialization
            policy = torch.load(self.model_path, map_location=self.device)
            if self.is_self_play:
                assert len(policy.actors) == 2, 'Separate-model self-play policy should have exactly 2 actors'
                self.actor: LatentActor = policy.actors[self.agent_id]
            else:
                assert len(policy.actors) == 1, 'Joint policy should have exactly 1 actor'
                self.actor: LatentActor = policy.actors[0]
            if policy.is_recurrent:
                self.rnn_hidden_dim = policy.rnn_hidden_dim
                self.rnn_states = torch.zeros(self.batch_size, self.rnn_hidden_dim, device=self.device)
            if not hasattr(self.actor, 'rnn'):
                self.actor.rnn = None

        if self.rnn_states is not None:
            self.rnn_states.zero_()


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


# def get_train_eval_pool(args):
#     assert args.env_name == 'Overcooked'
#     if args.desire_id is not None:
#         assert args.desire_id < 2 ** 5, f'Desire id out of range: {args.desire_id}'
#         policy_pool_train = [[((args.desire_id >> i) & 1) for i in range(5)]]
#         policy_pool_eval = []
#         self_play_opponents = 0
#         print('Put 1 desire into train pool')
#     elif args.rule_based_opponents > 0 or args.eval_pool_size > 0:
#         with open(args.env_config, 'r') as env_config_file:
#             env_map = yaml.safe_load(env_config_file)['mode']
#         print('Using map', env_map, 'and recipe type', args.recipe_type)
#         policy_pool_train_eval = generate_policy_pool(args.multi_agent > 1, args.p, env_map,
#                                                       args.rule_based_opponents + args.eval_pool_size,
#                                                       args.recipe_type, args.pool_seed)
#         policy_pool_train = policy_pool_train_eval[:args.rule_based_opponents]
#         policy_pool_eval = policy_pool_train_eval[args.rule_based_opponents:]
#         self_play_opponents = args.train_pool_size - args.rule_based_opponents
#         print('Put', len(policy_pool_train), 'rule-based opponents into train pool, ingredient support sets:',
#               [p.ingredient_support_set for p in policy_pool_train])
#         print('Put', len(policy_pool_eval), 'rule-based opponents into eval pool, ingredient support sets:',
#               [p.ingredient_support_set for p in policy_pool_eval])
#     else:
#         policy_pool_train = []
#         policy_pool_eval = []
#         self_play_opponents = args.train_pool_size
#     if self_play_opponents > 0:
#         assert 'potato_hard' in args.env_config, f'Loading potato hard fcp checkpoints for map {args.env_config}'
#         self_play_pool = load_potato_hard_self_play_policy_pool(1 - args.player_id)
#         assert len(self_play_pool) >= self_play_opponents, \
#             f'Requesting {self_play_opponents} self-play opponents, got {len(self_play_pool)}'
#         policy_pool_train.extend(self_play_pool[:self_play_opponents])
#         print('Put', self_play_opponents, 'self-play opponents into train pool, model paths:',
#               [p.model_path for p in self_play_pool[:self_play_opponents]])
#     return policy_pool_train, policy_pool_eval


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
    # print(ingredient_sets_all)
    ingredient_set_train = ingredient_sets_all[0:2]
    ingredient_set_eval = ingredient_sets_all[2:3]
    print(ingredient_set_train)
    print(ingredient_set_eval)
    i = 0
    j = 0
    while i < args.train_pool_size:
        for recipe in ingredient_set_train:
            policy = RuleBasedPolicy('minimum', np.random.rand() * args.p, 0, 0.1 * np.random.rand(), None,
                                     env_map, ingredient_support_set=recipe)
            policy_pool_train.append(policy)
            i = i + 1
    while j < args.eval_pool_size:
        for recipe in ingredient_set_eval:
            policy = RuleBasedPolicy('minimum', np.random.rand() * args.p, 0, 0.1 * np.random.rand(), None,
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
def generate_policy_pool(gen_desire, p_max, env_name, pool_size, recipe_type, pool_seed=1):
    old_state = np.random.get_state()
    np.random.seed(pool_seed)

    left_ingred = ingredients[:3]
    right_ingred = ingredients[3:]
    ingredient_sets_all = []
    # for i in range(1, 2 ** len(ingredients)):
    #     if i > (i & -i):
    #         ingredient_sets_all.append([ingredients[j] for j in range(len(ingredients)) if (i >> j) & 1])

    if recipe_type == 'full':
        # one ingredient in left
        for ingred in left_ingred:
            ingredient_sets_all.append([ingred])
        # two ingredient in left
        for i in range(len(left_ingred)):
            for j in range(i+1, len(left_ingred)):
                ingredient_sets_all.append([left_ingred[i], left_ingred[j]])
    else:
        assert recipe_type == 'cross', f'Unknown recipe type {recipe_type}'

    # one ingredient in left, one in right
    for ingred in left_ingred:
        for ingred2 in right_ingred:
            ingredient_sets_all.append([ingred, ingred2])
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


def load_good_self_play_policy_pool(player_id):
    print('Loading pretrained good self-play opponents...')
    good_policies = []
    good_policies2 = []
    for i in range(1, 16):
        run_path = './data/Overcooked/fcp_checkpoints/'
        good_policies.append(PretrainedPolicy(os.path.join(run_path, f'{i}_final.pt'), player_id))
        good_policies2.append(PretrainedPolicy(os.path.join(run_path, f'{i}_run2_final.pt'), player_id))
    print(f'{len(good_policies + good_policies2)} policies loaded.')
    # Fix partition across runs
    old_state = random.getstate()
    random.seed(1)
    random.shuffle(good_policies)
    random.shuffle(good_policies2)
    random.setstate(old_state)
    return good_policies + good_policies2[:10], good_policies2[10:]


def load_potato_self_play_policy_pool(player_id):
    print('Loading pretrained good self-play opponents on the potato map...')
    good_policies = []
    for i in range(12):
        run_path = './data/Overcooked/fcp_checkpoints/'
        good_policies.append(PretrainedPolicy(os.path.join(run_path, f'rule_potato_{i}_final.pt'), player_id))
    print(f'{len(good_policies)} policies loaded.')
    # Fix partition across runs
    old_state = random.getstate()
    random.seed(1)
    random.shuffle(good_policies)
    random.setstate(old_state)
    return good_policies


def load_potato_hard_self_play_policy_pool(player_id):
    print('Loading pretrained good self-play opponents on the potato hard map...')
    good_policies = []
    for i in range(12):
        run_path = './data/Overcooked/fcp_checkpoints/'
        good_policies.append(PretrainedPolicy(os.path.join(run_path, f'rule_potato_{i}_final.pt'), player_id))
    print(f'{len(good_policies)} policies loaded.')
    # Fix partition across runs
    old_state = random.getstate()
    random.seed(1)
    random.shuffle(good_policies)
    random.setstate(old_state)
    return good_policies


# TODO: pick a test policy pool
test_policies = None
