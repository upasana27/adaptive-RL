from gym_cooking.environment.game.game import Game

from gym_cooking.environment import cooking_zoo
from environment.overcooked.policy import RuleBasedPolicy
import numpy as np
n_agents = 2
num_humans = 1
max_steps = 100
render = False
p_max = 1.0
env_name = "1to1_env_3_divider"
level = '1to1_env_3_divider' # 'open_room_salad_easy' 
seed = 3
record = True
max_num_timesteps = 1000
recipes = [ "LettuceOnionSalad",
            "TomatoCarrotSalad",
            "PotatoBroccoliSalad"
            ]
ingredients = ['Lettuce','Tomato','Potato','Onion','Carrot','Broccoli']
Ingred2ID_desire = {
    0 : 0,
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 0,
    5: 0
}
parallel_env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record,
                                        max_steps=max_num_timesteps, recipes=recipes, desire = Ingred2ID_desire, obs_spaces=["dense"], obs_range = 10,
                                        interact_reward=0.5, progress_reward=1.0, complete_reward=10.0,
                                        step_cost=0.05)

action_spaces = parallel_env.action_spaces


class CookingAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation) -> int:
        return self.action_space.sample()

player_2_action_space = action_spaces["player_1"]
print(player_2_action_space)
cooking_agent = CookingAgent(player_2_action_space)

# define three rule based agents here
left_ingred = ingredients[:3]
right_ingred = ingredients[3:]
ingredient_sets_all = []
for i in (0,len(recipes)-1,1):
    ingredient_sets_all.append([left_ingred[i], right_ingred[i]])
print('All ingredient support sets:', ingredient_sets_all)
for ingredient_support_set in ingredient_sets_all:
    policy = RuleBasedPolicy('minimum', np.random.rand() * p_max, 0, 0.1 * np.random.rand(), None,
                                     env_name, ingredient_support_set=ingredient_support_set)
    print("created policy for support set : ", ingredient_support_set)

game = Game(parallel_env, num_humans, [policy], max_steps, render=True)
store = game.on_execute()

# game = Game(parallel_env, num_humans, [], max_steps, render=False)
# store = game.on_execute()

# game = Game(parallel_env, 0, [cooking_agent,cooking_agent], max_steps)
# store = game.on_execute_ai_only_with_delay()

print("done")
