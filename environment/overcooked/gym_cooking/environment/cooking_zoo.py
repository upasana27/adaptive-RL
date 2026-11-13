# Other core modules
import copy

from gym_cooking.cooking_world.cooking_world import CookingWorld
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_world.abstract_classes import *
from gym_cooking.cooking_book.recipe_drawer import RECIPES, NUM_GOALS

import numpy as np
from collections import namedtuple, defaultdict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import gym


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")
COLORS = ['blue', 'magenta', 'yellow', 'green']


Ingred2ID = {
    "Onion": 0,
    "Tomato": 1,
    "Lettuce": 2,
    "Carrot": 3,
    "Potato": 4,
    "Broccoli": 5
}

Ingred2ID_6recipe = {
    **Ingred2ID,
    "Pepper": 6,
    "Mushroom": 7,
    "Garlic": 8,
    "Cucumber": 9,
    "Spinach": 10,
    "Pumpkin": 11,
}


def env(*, level, num_agents, record, max_steps, recipes, desire, obs_spaces, obs_range,
        interact_reward, progress_reward, complete_reward,
        step_cost, recipe_domain='3recipe', board_placement_reward=0.0, board_chop_reward=0.0,
        plating_reward=0.0, plate_mixing='legacy'):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = CookingEnvironment(level, num_agents, record, max_steps,
                                  recipes, desire, obs_spaces, obs_range,
                                  interact_reward, progress_reward, complete_reward,
                                  step_cost, recipe_domain=recipe_domain,
                                  board_placement_reward=board_placement_reward,
                                  board_chop_reward=board_chop_reward,
                                  plating_reward=plating_reward, plate_mixing=plate_mixing)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class CookingEnvironment(AECEnv):
    """Environment object for Overcooked."""

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    def __init__(self, level, num_agents, record, max_steps, recipes, desire, obs_spaces, obs_range,
                 interact_reward=0.5, progress_reward=1.0, complete_reward=10.0,
                 step_cost=0.1, recipe_domain='3recipe', board_placement_reward=0.0, board_chop_reward=0.0,
                 plating_reward=0.0, plate_mixing='legacy'):
        super().__init__()
        # plate_mixing: 'legacy' (default, original cross-mixing of the R1-R3 ingredients) or 'pair'
        # (every ingredient accepts only its recipe partner; see cooking_world.PAIR_ONLY_MIXABLE).
        # 'shared_recipe' (2026-10-28, 6recipe_x4): a plate accepts a food iff some recipe in this env contains
        # everything on the plate plus the food (ingredients are shared across recipes in that domain).
        assert plate_mixing in ('legacy', 'pair', 'shared_recipe'), plate_mixing
        self.plate_mixing = plate_mixing
        self._mixing_sets = None
        # plating_reward: paid ONCE PER EPISODE on agent 0's FIRST plating of a chopped ingredient.
        self.plating_reward = float(plating_reward)
        # Optional L2 shaping (2026-10-21, default 0.0 = off): paid once per ingredient TYPE per
        # episode when agent 0 (the LEFT / L2 seat) places a RAW ingredient on the cutboard on
        # ITS OWN side. Added AFTER the max over recipe graphs, shared by both agents. Detected
        # as a state transition on the held object's identity, so a simultaneous interact by the
        # other agent cannot be mis-credited. See _board_placement_bonus().
        self.board_placement_reward = float(board_placement_reward)
        # Second optional knob: paid once per ingredient TYPE per episode when agent 0 completes a chop
        # on its own board (raw -> chopped transition caused by its interact). Same sharing/placement rules.
        self.board_chop_reward = float(board_chop_reward)
        self._board_bonus_types = set()
        self._chop_bonus_types = set()
        self._ep_board_placements = 0
        self._ep_l2_chops = 0
        self._ep_l2_retrievals = 0     # agent 0 picks a CHOPPED item up off its own board
        self._ep_l2_platings = 0       # agent 0 puts a CHOPPED item onto the plate
        self._ep_shaping = 0.0
        self._ep_env_return = 0.0
        # Which seat the shaping/counters above track. Default 0 (= the historical "agent 0 is the L2" assumption).
        # --both-sides runs set it per env process via OvercookedSingle.set_id so odd processes shape the ego, not the partner.
        self.shaping_agent_idx = 0

        self.recipe_domain = recipe_domain
        # '6recipe_x4' (2026-10-28): same 12 ingredients / classes / obs layout as '6recipe'; only the
        # recipe list (4-ingredient salads) and plate mixing differ.
        if recipe_domain in ('6recipe', '6recipe_x4'):
            self._ingred_2_id = Ingred2ID_6recipe
            self._game_classes = EXTENDED_GAME_CLASSES_6RECIPE
            self._game_classes_state_length = EXTENDED_GAME_CLASSES_STATE_LENGTH_6RECIPE
            self._holdable_idx = EXTENDED_GAME_CLASSES_HOLDABLE_IDX_6RECIPE
            self._food_classes = EXTENDED_FOOD_CLASSES_6RECIPE
            self._food_classes_idx = EXTENDED_FOOD_CLASSES_IDX_6RECIPE
            self._obj_idx = EXTENDED_OBJ_IDX_6RECIPE
            self._class_to_string = EXTENDED_CLASS_TO_STRING_6RECIPE
        else:
            self._ingred_2_id = Ingred2ID
            self._game_classes = GAME_CLASSES
            self._game_classes_state_length = GAME_CLASSES_STATE_LENGTH
            self._holdable_idx = GAME_CLASSES_HOLDABLE_IDX
            self._food_classes = FOOD_CLASSES
            self._food_classes_idx = FOOD_CLASSES_IDX
            self._obj_idx = OBJ_IDX
            self._class_to_string = ClassToString

        self.allowed_obs_spaces = ["symbolic", "numeric", "dense", "dense_partial", "2d", "2d_flatten"]
        assert len(obs_spaces)==1
        assert len(set(obs_spaces + self.allowed_obs_spaces)) == len(self.allowed_obs_spaces), \
            f"Selected invalid obs spaces. Allowed {self.allowed_obs_spaces}"
        assert len(obs_spaces) != 0, f"Please select an observation space from: {self.allowed_obs_spaces}"
        self.obs_spaces = obs_spaces
        if "dense_partial" in self.obs_spaces:
            assert obs_range is not None, f"Please identify the observation range if the observation space is dense partial"
            self.obs_range = obs_range
        else:
            self.obs_range = None
        # self.allowed_objects = allowed_objects or []
        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.level = level
        self.record = record
        self.max_steps = max_steps
        self.interact_reward = interact_reward
        self.progress_reward = progress_reward
        self.complete_reward = complete_reward
        self.desire = desire
        self.step_cost = step_cost
        print(f'Interact reward: {self.interact_reward}')
        print(f'Progress reward: {self.progress_reward}')
        print(f'Complete reward: {self.complete_reward}')
        print(f'Step cost: {self.step_cost}')
        self.t = 0
        self.filename = ""
        self.set_filename()
        self.world = CookingWorld(pair_only_mixing=(self.plate_mixing == 'pair'))
        self.recipes = recipes
        self.game = None
        self.recipe_graphs = [RECIPES[recipe]() for recipe in recipes]
        # sort the recipe so that the longest recipe will be checked for completion first
        self.recipe_graphs = sorted(self.recipe_graphs, key=lambda x: -len(x.node_list))
        # for recipe in self.recipe_graphs:
        #     print(recipe.name, [r.name for r in recipe.node_list])
        self.objects_in_recipes = set()
        self.recipe_desire = []
        for recipe in self.recipe_graphs:
            weight = 1
            for node in recipe.node_list:
                self.objects_in_recipes.add(node.root_type.__name__)
                if node.root_type.__name__ in self._ingred_2_id.keys():
                    weight = min(weight, desire[self._ingred_2_id[node.root_type.__name__]])
            self.recipe_desire.append(weight)
        print(f'Objects in recipes: {self.objects_in_recipes}')
        if self.plate_mixing == 'shared_recipe':
            self._mixing_sets = [frozenset(n.root_type.__name__ for n in r.node_list
                                           if n.root_type.__name__ in self._ingred_2_id)
                                 for r in self.recipe_graphs]
            self.world.mixing_sets = self._mixing_sets
            print(f'Plate mixing: shared_recipe over {len(self._mixing_sets)} recipes')
        self.termination_info = ""
        self.world.load_level(level=self.level, num_agents=num_agents)
        self.init_world_objs = copy.deepcopy(self.world.world_objects)
        self.graph_representation_length = sum([tup[1] for tup in self._game_classes_state_length]) + self.num_agents
        self.has_reset = True

        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        if "dense" in self.obs_spaces or "dense_partial" in self.obs_spaces:
            self.obs_size = self.get_obs_size(self.obs_spaces)
            self.observation_spaces = {agent: gym.spaces.Box(low=-1, high=1, shape=self.obs_size) for agent in self.possible_agents}
        elif "2d_flatten" in self.obs_spaces:
            self.obs_size = (self.world.width * self.world.height * (self.graph_representation_length), )
            self.observation_spaces = {agent: gym.spaces.Box(low=0, high=10, shape=self.obs_size) for agent in self.possible_agents}
        elif "2d" in self.obs_spaces:
            self.obs_size = (self.graph_representation_length, self.world.width, self.world.height)
            self.observation_spaces = {agent: gym.spaces.Box(low=0, high=10, shape=self.obs_size)
                                                                            for agent in self.possible_agents}
        
        self.action_spaces = {agent: gym.spaces.Discrete(6) for agent in self.possible_agents}
        self.held_obj = []
    
    def update_desire(self, desire):
        self.desire = desire
        assert all(d == 1 or d == 0 for d in desire), f'Desire error: {desire}'
        assert len(desire) == len(self._ingred_2_id.keys())
        for idx,recipe in enumerate(self.recipe_graphs):
            weight = 1
            for node in recipe.node_list:
                self.objects_in_recipes.add(node.root_type.__name__)
                if node.root_type.__name__ in self._ingred_2_id.keys():
                    weight = min(weight, self.desire[self._ingred_2_id[node.root_type.__name__]])
            self.recipe_desire[idx] = weight
        desire_set = [ig for ig in self._ingred_2_id.keys() if self.desire[self._ingred_2_id[ig]] == 1]
        print('Desire set to', desire_set)
        print('Recipe desire:', self.recipe_desire)
    
    def get_obs_size(self, obs_space):
        all_objs = copy.deepcopy(self.world.world_objects)
        
        n_obj = 0
        n_state = 0
        for obj_name, obj_list in all_objs.items():
            for obj in obj_list:
                if (not isinstance(obj, (Floor, Counter))):
                    n_obj += 1
                    if isinstance(obj, Food):
                        n_state += 1        
        if "dense" in self.obs_spaces:
            # agent pos + one-hot dir = 2 + 4 = 6
            # object pos = 2
            # All Food object done = 1 * (n_state)
            # next to counter in 4 direction = 4
            # which object is being held = len(GAME_CLASSES_HOLDABLE_IDX)
            # whether the front object can be interacted with = len(GAME_CLASSES) - 1, ignoring FLOOR object since it can never be used
            # whether the front object(food) is done = len(FOOD_CLASSES)
            # whether the holding object(food) is done = len(FOOD_CLASSES)
            obs_size = (6 * len(self.possible_agents) + 2 * n_obj + \
                 n_state + 4 + len(self._holdable_idx) + len(self._game_classes) - 1 + \
                 len(self._food_classes) + len(self._food_classes), )
        elif "dense_partial" in self.obs_spaces:
            # add a mask label for each part of obs, 0 for masked, 1 for seen
            # except next_to_counter & held_obj_idx/done & front_obj_interactable/done
            obs_size = (8 * len(self.possible_agents) + 3 * n_obj + \
                 2 * n_state + 4 + len(self._holdable_idx) + len(self._game_classes) - 1 + \
                 len(self._food_classes) + len(self._food_classes), )
            #print(f'agent:{len(self.possible_agents)}, obj:{n_obj}, food:{n_state}')
        else:
            raise NotImplementedError
        return obs_size
    
    def set_filename(self):
        self.filename = f"{self.level}_agents{self.num_agents}"

    def state(self):
        pass

    def reset(self):
        self.world = CookingWorld(pair_only_mixing=(self.plate_mixing == 'pair'), mixing_sets=self._mixing_sets)
        self.t = 0
        self.held_obj = []
        self._board_bonus_types = set()
        self._chop_bonus_types = set()
        self._ep_board_placements = 0
        self._ep_l2_chops = 0
        self._ep_l2_retrievals = 0
        self._ep_l2_platings = 0
        self._ep_shaping = 0.0
        self._ep_env_return = 0.0

        # For tracking data during an episode.
        self.termination_info = ""

        # Load world & distances.
        self.world.load_level(level=self.level, num_agents=self.num_agents)
        for obj1,obj2 in zip(self.init_world_objs, self.world.world_objects):
            assert type(obj1)==type(obj2)

        for recipe in self.recipe_graphs:
            recipe.update_recipe_state(self.world)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))

        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []

    def close(self):
        return

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    _DIVIDER_X = 3   # both the FC and FI maps split the kitchen at x=3 (RuleBasedPolicy.divider_loc)

    def _left_boards(self, agent):
        """Cutboards on the same side of the divider as `agent` (the only ones it can reach)."""
        side = agent.location[0] < self._DIVIDER_X
        return [cb for cb in self.world.world_objects['CutBoard'] if (cb.location[0] < self._DIVIDER_X) == side]

    def _board_snapshot(self):
        """Pre-step state needed to detect agent-0 board placements and chops as transitions."""
        a0 = self.world.agents[self.shaping_agent_idx]
        boards = self._left_boards(a0)
        foods_done = {}
        for cb in boards:
            for f in self.world.get_objects_at(cb.location, Food):
                foods_done[id(f)] = f.done()
        return a0.holding, boards, foods_done

    def _board_placement_bonus(self, snap, a0_action=None):
        """Return the shaping bonus for this step and update per-episode counters."""
        held_before, boards, foods_done_before = snap
        a0 = self.world.agents[self.shaping_agent_idx]
        bonus = 0.0
        # retrieval: agent 0 had empty hands and now holds a CHOPPED food that was on its board
        if (held_before is None and a0.holding is not None and isinstance(a0.holding, Food)
                and a0.holding.done() and id(a0.holding) in foods_done_before):
            self._ep_l2_retrievals += 1
        # plating: the CHOPPED food agent 0 was holding is now inside a plate
        if (held_before is not None and isinstance(held_before, Food) and held_before.done()
                and a0.holding is None
                and any(held_before in getattr(p, 'content', []) for p in self.world.world_objects['Plate'])):
            self._ep_l2_platings += 1
            if self.plating_reward and self._ep_l2_platings == 1:
                bonus += self.plating_reward
        # placement: the exact object agent 0 was holding is now on one of its boards, raw, and
        # its hands are empty. Object identity makes this immune to the other agent's actions.
        if (held_before is not None and isinstance(held_before, Food) and not held_before.done()
                and a0.holding is None
                and any(held_before.location == cb.location for cb in boards)):
            self._ep_board_placements += 1
            tname = type(held_before).__name__
            if self.board_placement_reward and tname not in self._board_bonus_types:
                self._board_bonus_types.add(tname)
                bonus = self.board_placement_reward
        # chop: a food on agent 0's board went raw -> chopped this step. Only agent 0 can reach that
        # board, and we additionally require agent 0 to have pressed interact (5) this step.
        for cb in boards:
            for f in self.world.get_objects_at(cb.location, Food):
                if foods_done_before.get(id(f)) is False and f.done() and (a0_action is None or int(a0_action) == 5):
                    self._ep_l2_chops += 1
                    tname = type(f).__name__
                    if self.board_chop_reward and tname not in self._chop_bonus_types:
                        self._chop_bonus_types.add(tname)
                        bonus += self.board_chop_reward
        return bonus

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        board_snap = self._board_snapshot()
        self.world.perform_agent_actions(self.world.agents, actions)

        # Visualize.
        if self.record:
            self.game.on_render()

        if self.record:
            self.game.save_image_obs(self.t)

        for agent in self.agents:
            self.current_tensor_observation[agent] = self.get_tensor_representation(agent)

        done, rewards, goals = self.compute_rewards()
        info = {"t": self.t, "termination_info": self.termination_info}
        if self.obs_range is not None:
            info['keypoint_visited'] = [self.world_agent_mapping[agent].location[1] > self.obs_range for agent in self.agents]
        if self.termination_info.endswith('timesteps'):
            info['bad_transition'] = True
        env_r = max(rewards) - self.step_cost  # use the highest progressed recipe as a reward
        bonus = self._board_placement_bonus(board_snap, a0_action=actions[self.shaping_agent_idx] if len(actions) > self.shaping_agent_idx else None)
        self._ep_env_return += float(env_r)
        self._ep_shaping += float(bonus)
        info['l2_board_placements'] = self._ep_board_placements
        info['l2_chops'] = self._ep_l2_chops
        info['l2_retrievals'] = self._ep_l2_retrievals
        info['l2_platings'] = self._ep_l2_platings
        info['shaping_reward'] = self._ep_shaping
        info['env_return'] = self._ep_env_return
        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.rewards[agent] = env_r + bonus   # shaping added AFTER the max, shared identically
            self.infos[agent] = info

    def in_obs(self, agent, target_loc):
        self_loc = np.array(self.world_agent_mapping[agent].location)
        # obs range refers to the y-axis coordinate of the wall
        # can only see the objects on the same side of the wall

        #dist = np.sqrt(np.sum((self_loc-target_loc)**2))
        in_vis = (self_loc[1]<= self.obs_range and target_loc[1]<=self.obs_range) or (self_loc[1]> self.obs_range and target_loc[1]>self.obs_range)
        #print(self_loc, target_loc, in_vis)
        return in_vis

    def observe(self, agent):
        observation = []
        if "numeric" in self.obs_spaces:
            num_observation = {'numeric_observation': self.current_tensor_observation[agent],
                               'agent_location': np.asarray(self.world_agent_mapping[agent].location, np.int32),
                               'goal_vector': self.recipe_mapping[agent].goals_completed(NUM_GOALS)}
            observation.append(num_observation)
        if "symbolic" in self.obs_spaces:
            objects = defaultdict(list)
            objects.update(self.world.world_objects)
            objects["Agent"] = self.world.agents
            sym_observation = copy.deepcopy(objects)
            observation.append(sym_observation)
        if "2d" in self.obs_spaces:
            obs = self.current_tensor_observation[agent].transpose(2,1,0)
            observation.append(obs)
        if "2d_flatten" in self.obs_spaces:
            obs = self.current_tensor_observation[agent]
            obs = obs.reshape(-1)
            observation.append(obs)

        ### Add numeric representation for symbolic data
        if "dense" in self.obs_spaces or "dense_partial" in self.obs_spaces:
            # pass
            # all objects [Counter, Floor, *static_objs, *dynamic_objs, *agents]
            dense_obs = []
            world_shape = np.array([self.world.width, self.world.height], np.float32)
            all_objs = self.world.world_objects # self.all_objs
            
            state_feat = []
            next_to_counter_feat = np.zeros(4)
            pos_feat = []
            dir_feat = []
            in_obs_mask = []
            agent_obj = self.world_agent_mapping[agent]
            own_pos = np.asarray(agent_obj.location)
            eye = np.eye(4)
            dir_feat.append(eye[agent_obj.orientation-1])
            pos_feat.append(own_pos)
            in_obs_mask.append(True)
            # other agents' pos
            for a in self.possible_agents:
                if a != agent:
                    other_pos = np.array(self.world_agent_mapping[a].location)
                    rel_pos = other_pos - own_pos
                    pos_feat.append(rel_pos)
                    dir_feat.append(eye[self.world_agent_mapping[a].orientation-1])
                    if "dense_partial" in self.obs_spaces:
                        in_obs_mask.append(self.in_obs(agent, other_pos))
            if "dense_partial" in self.obs_spaces:
                dir_mask = np.array(in_obs_mask).reshape(-1,1)
                dir_feat = np.array(dir_feat) * dir_mask 
                dir_feat = np.concatenate((dir_feat,dir_mask),-1)
                dir_feat = np.array(dir_feat).reshape(-1)
            else:
                dir_feat = np.array(dir_feat).reshape(-1)
            # static obj pos
            for obj_name, obj_list in all_objs.items():
                for obj in obj_list:
                    if (not isinstance(obj, Floor)):
                        if not isinstance(obj, Counter):
                            rel_pos = np.array(obj.location) - own_pos
                            pos_feat.append(rel_pos)
                            if "dense_partial" in self.obs_spaces:
                                in_obs_mask.append(self.in_obs(agent, np.array(obj.location)))
                        if isinstance(obj, Food):
                            if "dense_partial" in self.obs_spaces:
                                in_obs_bool = self.in_obs(agent, np.array(obj.location))
                                state_feat.append([obj.done()*in_obs_bool, in_obs_bool])
                            else:
                                state_feat.append(obj.done())
                        elif isinstance(obj, StaticObject):
                            if all(np.array(obj.location) == (own_pos + np.array([1,0]))):
                                next_to_counter_feat[0]=1
                            elif all(np.array(obj.location) == (own_pos + np.array([0,1]))):
                                next_to_counter_feat[1]=1
                            elif all(np.array(obj.location) == (own_pos + np.array([-1,0]))):
                                next_to_counter_feat[2]=1
                            elif all(np.array(obj.location) == (own_pos + np.array([0,-1]))):
                                next_to_counter_feat[3]=1
                         
            pos_feat = np.array(pos_feat, np.float32) / world_shape
            if "dense_partial" in self.obs_spaces:
                pos_mask = np.array(in_obs_mask).reshape(-1,1)
                pos_feat = np.array(pos_feat) * pos_mask 
                pos_feat = np.concatenate((pos_feat,pos_mask),-1)
                pos_feat = pos_feat.reshape(-1) 
            else:
                pos_feat = pos_feat.reshape(-1) 
            state_feat = np.array(state_feat).reshape(-1)
        
            # what is in the held obj
            holding_feat = np.zeros(len(self._holdable_idx))
            holding_state_feat = np.zeros(len(self._food_classes))
            held_obj = agent_obj.holding
            if held_obj is not None:
                held_obj_name = type(held_obj).__name__
                holding_feat[self._holdable_idx[held_obj_name]] = 1
                if isinstance(held_obj, Food):
                    holding_state_feat[self._food_classes_idx[held_obj_name]] = held_obj.done()
                if isinstance(held_obj, Container):
                    if held_obj.content:
                        for obj in held_obj.content:
                            obj_name = type(obj).__name__
                            holding_feat[self._holdable_idx[obj_name]] = 1
                            if isinstance(obj, Food):
                                holding_state_feat[self._food_classes_idx[obj_name]] = obj.done()

            # the cell in-front is interactable/pickable
            front_obj_feat = np.zeros(len(self._game_classes)-1)
            front_pos = self.world.get_target_location(agent_obj, agent_obj.orientation)
            front_obj_list = self.world.get_objects_at(front_pos)
            front_obj_state_feat = np.zeros(len(self._food_classes))
            for obj in front_obj_list:
                if not isinstance(obj, Floor):
                    obj_name = type(obj).__name__
                    front_obj_feat[self._obj_idx[obj_name]] = 1
                    if isinstance(obj, Food):
                        front_obj_state_feat[self._food_classes_idx[obj_name]] = obj.done()

            # Agents
            dense_obs = np.concatenate([pos_feat, dir_feat, state_feat,
                                       next_to_counter_feat,
                                       holding_feat, holding_state_feat,
                                       front_obj_feat, front_obj_state_feat])
            observation.append(dense_obs)
            
        returned_observation = observation if not len(observation) == 1 else observation[0]
        return returned_observation

    def compute_rewards(self):
        done = False
        rewards = np.zeros(len(self.recipes), dtype=np.float32) # [0] * len(self.recipes)
        open_goals = [[0]] * len(self.recipes)
        # Done if the episode maxes out
        if self.t >= self.max_steps and self.max_steps:
            self.termination_info = f"Terminating because passed {self.max_steps} timesteps"
            done = True

        for i,agent in enumerate(self.world.agents):
            if agent.holding is not None:
                state = [None]
                holding_obj = agent.holding
                if type(holding_obj).__name__ in self.objects_in_recipes:
                    weight = 1
                    if isinstance(agent.holding , Food):
                        state = [agent.holding.done()]
                        if type(holding_obj).__name__ in self._ingred_2_id.keys():
                            weight = self.desire[self._ingred_2_id[type(holding_obj).__name__]]
                    elif isinstance(agent.holding, Container):
                        if agent.holding.content:
                            food_names = [type(food).__name__ for food in agent.holding.content]
                            state = food_names
                            weight = 1
                            for name in food_names:
                                if name in self._ingred_2_id.keys():
                                    weight = min(weight, self.desire[self._ingred_2_id[name]])

                    obj_with_state = {agent, holding_obj, *state}
                    if obj_with_state not in self.held_obj:
                        self.held_obj.append(obj_with_state)
                        rewards += self.interact_reward * weight

        for idx, recipe in enumerate(self.recipe_graphs):
            if not recipe.is_completable(self.world):
                continue
            goals_before = recipe.goals_completed(NUM_GOALS)
            recipe.update_recipe_state(self.world)
            open_goals[idx] = recipe.goals_completed(NUM_GOALS)
            bonus = recipe.completed() * self.complete_reward * self.recipe_desire[idx]
            rewards[idx] += bonus if bonus else ((sum(goals_before) - sum(open_goals[idx])) * self.progress_reward * self.recipe_desire[idx]) # * 5
            if rewards[idx] < 0:
                print(f"Goals before: {goals_before}")
                print(f"Goals after: {open_goals}")
        
        for recipe in self.recipe_graphs:
            if recipe.completed():
                self.termination_info = f"Terminating because {recipe.name} is completed"
                done = True
                break
        return done, rewards, open_goals

    def get_tensor_representation(self, agent):
        tensor = np.zeros(
            (self.world.width, self.world.height, self.graph_representation_length))
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        idx = 0
        for game_class in self._game_classes:
            if game_class is Agent:
                continue
            stateful_class = self.get_stateful_class(game_class)
            if stateful_class:
                n = 1
                for obj in objects[self._class_to_string[game_class]]:
                    representation = self.handle_stateful_class_representation(obj, stateful_class)
                    n = len(representation)
                    x, y = obj.location
                    for i in range(n):
                        tensor[x, y, idx + i] += representation[i]
                idx += n
            else:
                for obj in objects[ClassToString[game_class]]:
                    x, y = obj.location
                    tensor[x, y, idx] += 1
                idx += 1

        ego_agent = self.world_agent_mapping[agent]
        x, y = ego_agent.location
        # location map for all agents, location maps for separate agent and four orientation maps shared
        # between all agents
        tensor[x, y, idx] = 1
        tensor[x, y, idx + 1] = 1
        tensor[x, y, idx + self.num_agents + ego_agent.orientation] = 1

        agent_idx = 1
        for world_agent in self.world.agents:
            if agent != world_agent:
                x, y = world_agent.location
                # location map for all agents, location maps for separate agent and four orientation maps shared
                # between all agents
                tensor[x, y, idx] = 1
                tensor[x, y, idx + agent_idx + 1] = 1
                tensor[x, y, idx + self.num_agents + world_agent.orientation] = 1
                agent_idx += 1
        return tensor

    def get_agent_names(self):
        return [agent.name for agent in self.world.agents]

    def render(self, mode='human'):
        pass

    @staticmethod
    def get_stateful_class(game_class):
        for stateful_class in STATEFUL_GAME_CLASSES:
            if issubclass(game_class, stateful_class):
                return stateful_class
        return None

    @staticmethod
    def handle_stateful_class_representation(obj, stateful_class):
        if stateful_class is ChopFood:
            return [int(obj.chop_state == ChopFoodStates.CHOPPED)]
        if stateful_class is BlenderFood:
            return [obj.current_progress]
        raise ValueError(f"Could not process stateful class {stateful_class}")
