"""Rollout loading and event extraction shared by the behaviour analyses.

A rollout file (written by collect_rollouts.py) has one JSON object per environment step:
    env      id of the environment instance (one instance per partner)
    partner  index of the partner in the evaluation pool (absent in older files)
    ep, t    episode counter of that instance, step within the episode (from 1)
    ego      seat of the ego agent (0 = left, 1 = right)
    ev_ego, ev_par   events of the ego / partner at this step, e.g. "picked up Lettuce",
                     "chopped Lettuce", "put Lettuce on the Plate", "DELIVERED ..."
    hold_ego, hold_par, plate, pos_ego, pos_par, a_ego, a_par, rew, done
"""
import collections
import json
import re

RECIPE_PAIRS = {  # recipe -> (left ingredient, right ingredient)
    'FC': {'LO': ('Lettuce', 'Onion'), 'TC': ('Tomato', 'Carrot'), 'PB': ('Potato', 'Broccoli')},
    'FI': {'R1': ('Lettuce', 'Onion'), 'R2': ('Potato', 'Broccoli'), 'R3': ('Tomato', 'Carrot'),
           'R4': ('Cucumber', 'Pepper'), 'R5': ('Spinach', 'Mushroom'), 'R6': ('Pumpkin', 'Garlic')},
}
RECIPE_PAIRS['PC'] = RECIPE_PAIRS['FC']
FI_TYPE = {'R1': 'R1R2', 'R2': 'R1R2', 'R3': 'R3R4', 'R4': 'R3R4', 'R5': 'R5R6', 'R6': 'R5R6'}

_PICK = re.compile(r"picked up (\w+)$")          # raw ingredient taken from the wall (not a chopped one)
_PLATE = re.compile(r"put (\w+) on the Plate")


def ingredient_to_recipe(domain):
    return {ing: rec for rec, pair in RECIPE_PAIRS[domain].items() for ing in pair}


def load_episodes(paths, min_steps=2):
    """Return {(file index, env, ep): [step, ...]} with steps in time order."""
    episodes = collections.defaultdict(list)
    for i, path in enumerate(paths):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                episodes[(i, row['env'], row['ep'])].append(row)
    return {k: v for k, v in episodes.items() if v[-1]['t'] >= min_steps}


def parse_episode(steps, domain):
    """Summarise one episode.

    Returns a dict with
        seat        ego seat
        success     a dish was delivered
        ego_picks, partner_picks   [(t, recipe)] raw-ingredient pick-ups in order
        plated      [recipe] of every ingredient put on the plate, in order
        target      recipe of the last plated ingredient (the recipe the pair ended up preparing), or None
    """
    ing = ingredient_to_recipe(domain)

    def picks(key):
        out = []
        for s in steps:
            for e in s[key]:
                m = _PICK.match(e)
                if m and m.group(1) in ing:
                    out.append((s['t'], ing[m.group(1)]))
        return out

    plated = [ing[m.group(1)] for s in steps for e in s['ev_ego'] + s['ev_par']
              for m in [_PLATE.match(e)] if m and m.group(1) in ing]
    success = any('DELIVERED' in e for s in steps for e in s['ev_ego'] + s['ev_par'])
    return dict(seat=steps[0]['ego'], success=success, ego_picks=picks('ev_ego'), partner_picks=picks('ev_par'),
                plated=plated, target=plated[-1] if plated else None)


def rounds(episodes):
    """Group the episodes of each environment instance into rounds (the 5-episode interaction windows).

    The evaluation clears the ego's history at the end of every round and cuts the episode an instance is in
    at that moment. Unfinished episodes are dropped here, so a round boundary appears as a gap in the
    instance's episode counter. An instance that finishes an episode exactly at the boundary shows no gap and
    its next round is appended to the current one; report positions 1 to 5 only.
    Returns [(key prefix, [steps of episode at position 1, position 2, ...])].
    """
    by_env = collections.defaultdict(dict)
    for (i, env, ep), steps in episodes.items():
        if steps[-1]['done']:                       # unfinished episodes mark the round boundaries
            by_env[(i, env)][ep] = steps
    out = []
    for key, eps in by_env.items():
        cur, prev = [], None
        for ep in sorted(eps):
            if prev is not None and ep != prev + 1 and cur:
                out.append((key, cur)); cur = []
            cur.append(eps[ep]); prev = ep
        if cur:
            out.append((key, cur))
    return out
