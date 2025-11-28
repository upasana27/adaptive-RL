import sys
# sys.path.append("/home/asurite.ad.asu.edu/ubiswas2/adaptive-RL")
# print(sys.path)
from gym_cooking.environment.game.game import Game
import yaml
from gym_cooking.environment import cooking_zoo
from overcooked_maker import OvercookedMaker
from policy import RuleBasedPolicy
from policy import PretrainedPolicy
n_agents = 2
num_humans = 1
max_steps = 40
render = False

level = 'fc_small_divider_test' # 'open_room_salad_easy' 
seed = 3
record = False
max_num_timesteps = 100
env_config = "/home/asurite.ad.asu.edu/ubiswas2/adaptive-RL/environment/overcooked/config/fc_test.yaml"
Ingred2ID = {
    "Onion": 0,
    "Tomato": 1,
    "Lettuce": 2,
    "Carrot": 3,
    "Potato": 4,
    "Broccoli": 5
}
ingredients = ['Lettuce','Tomato','Potato','Onion','Carrot','Broccoli']
recipes= [
  "TomatoOnionSalad",
  "OnionTomatoSalad",
 "LettuceBroccoliSalad",
  "BroccoliLettuceSalad",
  "PotatoCarrotSalad",
  "CarrotPotatoSalad",
]
default_desire = [1] * len(Ingred2ID)
parallel_env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=False, 
                                        max_steps=max_num_timesteps, recipes=recipes, obs_spaces=["dense_partial"],desire=default_desire, obs_range=15,
                                        interact_reward=0.5, progress_reward=1.0,
                                        complete_reward=10.0, step_cost=0.05)

# action_spaces = parallel_env.action_spaces
# maker = OvercookedMaker(mode=level, horizon=max_steps, recipes=recipes, obs_spaces=["dense_partial"], obs_range=15, num_agents=n_agents,
#                         interact_reward=0.5, progress_reward=1.0, complete_reward=10.0,
#                         step_cost=0.1, display=False)

# class CookingAgent:

#     def __init__(self, action_space):
#         self.action_space = action_space

#     def get_action(self, observation) -> int:
#         return self.action_space.sample()

# player_2_action_space = action_spaces["player_1"]
# cooking_agent = CookingAgent(player_2_action_space)

pref_recipe = ["Lettuce","Broccoli"]
# with open(env_config, 'r') as env_config_file:
#         env_map = yaml.safe_load(env_config_file)['mode']
#         print('Using map', env_map, 'and recipe type')
# cooking_agent = RuleBasedPolicy('full', 0, 0, 0, None,env_map, ingredient_support_set=pref_recipe,agent_id=0)
model_path = "/home/asurite.ad.asu.edu/ubiswas2/Desktop/PACE/logs/Overcooked/ppo_pace_seed1/ppo/30000000.pt"
cooking_agent = PretrainedPolicy(level=level, num_agents=n_agents,max_steps=max_num_timesteps, recipes=recipes,record=False, obs_spaces=["dense_partial"],desire=default_desire, obs_range=15,
                interact_reward=0.5, progress_reward=1.0,
                complete_reward=10.0, step_cost=0.05,model_path=model_path, agent_id=0, device='cpu')
game = Game(parallel_env, num_humans, [cooking_agent], max_steps, render=False)
store = game.on_execute()

# game = Game(parallel_env, num_humans, [], max_steps, render=False)
# store = game.on_execute()

# game = Game(parallel_env, 0, [cooking_agent,cooking_agent], max_steps)
# store = game.on_execute_ai_only_with_delay()

print("done")