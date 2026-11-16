"""Run an evaluation (online_test_.py) and log the events of both agents at every step.

Usage:  python analysis/collect_rollouts.py <out.jsonl> <online_test_.py arguments ...>
or      SAVE_ROLLOUTS=1 bash scripts/overcooked/<DOMAIN>/<SIDE>/eval_method_vs_L1.sh   (writes <run>.jsonl next to the log)

The evaluation itself is unchanged: the environments are stepped in-process and the world state before and
after each step is compared to obtain pick-ups, put-downs, chops, plating and delivery. See events.py for
the format of the output file.
"""
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
OUT = open(sys.argv[1], "w")

import learning.envs as E
from stable_baselines3.common.vec_env import DummyVecEnv
E.SubprocVecEnv = DummyVecEnv          # in-process environments so the steps can be observed

from environment.overcooked import overcooked_single as OS
from llm_agent import actions as A
from llm_agent.memory import snapshot, step_events

_step, _reset, _set_opp = OS.Overcooked_SingleEnv.step, OS.Overcooked_SingleEnv.reset, OS.Overcooked_SingleEnv.set_opponent
_state, _n_partners = {}, [0]


def _holding(world, i):
    h = getattr(world.agents[i], "holding", None)
    if h is None:
        return None
    name = type(h).__name__
    if name == "Plate":
        return "Plate" + str([type(c).__name__ for c in getattr(h, "content", [])])
    return name + ("*" if h.done() else "")


def _plate(world):
    plates = world.world_objects.get("Plate", [])
    return [type(c).__name__ for c in getattr(plates[0], "content", [])] if plates else None


def set_opponent(self, policy):
    st = _state.setdefault(id(self), {"ep": -1, "t": 0, "partner": None})
    if st["partner"] is None:              # partners are assigned once, in pool order
        st["partner"] = _n_partners[0]; _n_partners[0] += 1
    return _set_opp(self, policy)


def reset(self):
    st = _state.setdefault(id(self), {"ep": -1, "t": 0, "partner": None})
    st["ep"] += 1; st["t"] = 0
    return _reset(self)


def step(self, action):
    world = self.env.get_env_world(); before = snapshot(world); me = self._id
    obs, rew, done, info = _step(self, action)
    world = self.env.get_env_world()
    acts = {me: int(action), 1 - me: int(info.get("opponent_act", -1))}
    ev = step_events(before, world, acts, bool(rew >= 9))
    st = _state[id(self)]; st["t"] += 1
    OUT.write(json.dumps(dict(
        env=id(self) % 100000, partner=st["partner"], ep=st["ep"], t=st["t"], ego=me,
        a_ego=A.NAMES.get(acts[me]), a_par=A.NAMES.get(acts[1 - me]), ev_ego=ev[me], ev_par=ev[1 - me],
        rew=float(rew), done=bool(done), hold_ego=_holding(world, me), hold_par=_holding(world, 1 - me),
        plate=_plate(world), pos_ego=list(world.agents[me].location), pos_par=list(world.agents[1 - me].location))) + "\n")
    if done:
        OUT.flush()
    return obs, rew, done, info


OS.Overcooked_SingleEnv.step, OS.Overcooked_SingleEnv.reset, OS.Overcooked_SingleEnv.set_opponent = step, reset, set_opponent
sys.argv = ["online_test_.py"] + sys.argv[2:]
runpy.run_path("online_test_.py", run_name="__main__")
