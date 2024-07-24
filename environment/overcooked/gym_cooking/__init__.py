from gym.envs.registration import register
from rebar import dotdict, arrdict

Dotdict = dotdict.dotdict
Arrdict = arrdict.arrdict

register(id="cookingEnv-v1",
         entry_point="gym_cooking.environment:GymCookingEnvironment")
register(id="cookingZooEnv-v0",
         entry_point="gym_cooking.environment:CookingZooEnvironment")
