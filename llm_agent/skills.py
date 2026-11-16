"""High-level skill controller for the LLM agent (ProAgent-style).

Modeled on ProAgent (Zhang et al., AAAI 2024; github.com/PKU-Alignment/ProAgent):
  * the LLM picks ONE skill from a fixed set whose PRECONDITIONS are given as pseudo-code;
  * a controller turns the skill into low-level moves (BFS over floor, re-planned every
    step, one env action per step) and checks a per-skill DONE condition;
  * preconditions are re-validated every step; on failure the LLM is told which
    assertion failed and asked to re-plan (max 3 in a row, then wait(1)).
    ProAgent asks a second LLM ("explainer") to say which assertion failed; here the same
    information is produced deterministically from the world state, which is strictly
    more reliable and saves a call.
  * the LLM is queried only when a skill completes or fails (ProAgent's scene loop).
  * belief tracking: the LLM's predicted partner intention at each decision is kept next to
    the partner's OBSERVED high-level events (ProAgent reads the partner's true skill labels
    from Overcooked-AI; this env has none, so observed events stand in).

Skill set for THIS game (two-sided kitchen, one Plate, chop/plate irreversible):
  pickup(X)          place_on_counter()   chop()        put_on_plate()
  pickup_plate()     deliver()            wait(n)
The ProAgent "place_obj_on_counter" and "deliver" semantics are kept; chop() is new because
this game chops on a CutBoard instead of cooking in a pot.
"""
import random
import re
from collections import deque

from llm_agent import actions as A
from llm_agent.state_text import CHOPPED, _on_plate

SKILL_NAMES = ("pickup", "place_on_counter", "chop", "put_on_plate", "pickup_plate", "deliver", "wait")
MAX_WAIT = 10
MAX_REPLANS = 3
STUCK_LIMIT = 3          # steps without position change before a random unstick move


# ---------------------------------------------------------------- world helpers
def _floor(world):
    return {tuple(o.location) for o in world.world_objects.get("Floor", [])}


def _bfs(world, start, blocked=()):
    """{cell: prev} shortest-path tree over floor from start, avoiding `blocked` cells."""
    floor = _floor(world) - set(blocked)
    start = tuple(start)
    prev = {start: None}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n in floor and n not in prev:
                prev[n] = (x, y)
                q.append(n)
    return prev


def _path(prev, goal):
    if goal not in prev:
        return None
    p, c = [], goal
    while c is not None:
        p.append(c)
        c = prev[c]
    return p[::-1]          # start .. goal


def _dir_to(frm, to):
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    return {(-1, 0): A.LEFT, (1, 0): A.RIGHT, (0, 1): A.DOWN, (0, -1): A.UP}[(dx, dy)]


def _held(world, me_id):
    return getattr(world.agents[me_id], "holding", None)


def _is_chopped(o):
    return CHOPPED is not None and getattr(o, "chop_state", None) == CHOPPED


def _plate(world):
    ps = world.world_objects.get("Plate", [])
    return ps[0] if ps else None


def _loose_items(world):
    """Ingredient objects lying on the map (not held, not on the plate)."""
    out = []
    for cls, objs in world.world_objects.items():
        if cls in ("Counter", "Floor", "CutBoard", "DeliverSquare", "Blender", "Plate"):
            continue
        for o in objs:
            if not hasattr(o, "location"):
                continue
            if any(getattr(a, "holding", None) is o for a in world.agents) or _on_plate(world, o):
                continue
            out.append(o)
    return out


def _occupied_cells(world):
    """Cells that currently hold a loose item or the plate (a counter with something on it)."""
    cells = {tuple(o.location) for o in _loose_items(world)}
    p = _plate(world)
    if p is not None and not any(getattr(a, "holding", None) is p for a in world.agents):
        cells.add(tuple(p.location))
    return cells


def stand_options(world, me_id, target, blocked=()):
    """Reachable (stand_cell, facing_action, path) triples for acting on `target`, nearest first."""
    me = world.agents[me_id]
    prev = _bfs(world, me.location, blocked)
    opts = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        stand = (target[0] - dx, target[1] - dy)
        if stand in prev:
            path = _path(prev, stand)
            opts.append((len(path), stand, _dir_to(stand, target), path))
    opts.sort(key=lambda x: x[0])
    return [(s, f, p) for _, s, f, p in opts]


def reachable_targets(world, me_id, kind):
    """Locations of `kind` ('CutBoard'/'DeliverSquare'/'Counter') I can act on from my side."""
    out = []
    for o in world.world_objects.get(kind, []):
        loc = tuple(o.location)
        if stand_options(world, me_id, loc):
            out.append(loc)
    return out


def side_of(world, me_id, loc):
    """'yours' if agent me_id can act on loc from its floor region, else 'partner'."""
    return "yours" if stand_options(world, me_id, tuple(loc)) else "partner"


# ---------------------------------------------------------------- skill parsing
_SKILL_RE = re.compile(r"\b(pickup_plate|pickup|place_on_counter|put_on_plate|chop|deliver|wait)\s*\(\s*([A-Za-z0-9_ ]*)\s*\)")


def parse_skill(text, field="Plan"):
    """Return (name, arg) from the LAST '<field>: ...' line containing a skill call, else None."""
    if not text:
        return None
    lines = [l for l in text.splitlines() if re.match(rf"\s*\**{field}\**\s*:", l, flags=re.I)]
    cand = lines[-1] if lines else text
    m = _SKILL_RE.findall(cand)
    if not m and lines:
        return None
    if not m:
        return None
    name, arg = m[-1]
    arg = arg.strip().strip('"\'')
    return name, (arg or None)


def skill_str(name, arg):
    return f"{name}({arg if arg is not None else ''})"


# ---------------------------------------------------------------- controller
class SkillController:
    """Executes one skill over several env steps. Call .plan(skill) then .step(world) per step.

    step() returns (low_level_action, status) with status in
      'running' | 'done' | 'failed:<reason>'.
    """

    def __init__(self, me_id, recipes_ingredients, rng=None):
        self.me = me_id
        self.ingredients = set(recipes_ingredients)
        self.rng = rng or random.Random(0)
        self.skill = None
        self.steps = 0
        self.stuck = 0
        self.last_pos = None
        self.wait_left = 0
        self._phase = None          # for chop(): 'put' -> 'chop' -> 'take'
        self._board = None          # CutBoard location chosen for chop()
        self._item = None           # object being chopped / placed

    # ---- preconditions (ProAgent: assert-style, reported verbatim on failure) ----
    def check(self, world, name, arg):
        """Return None if preconditions hold, else the failed assertion as text."""
        held = _held(world, self.me)
        hn = type(held).__name__ if held is not None else None
        if name == "pickup":
            if arg is None or arg not in self.ingredients:
                return f"pickup({arg}): {arg!r} is not an ingredient of this kitchen"
            if held is not None:
                return f"pickup({arg}): your hand is not empty (you hold {hn})"
            objs = [o for o in _loose_items(world) if type(o).__name__ == arg]
            if not objs:
                return f"pickup({arg}): no loose {arg} on the map (held, plated, or absent)"
            if not any(stand_options(world, self.me, tuple(o.location)) for o in objs):
                return f"pickup({arg}): {arg} is not reachable from your side"
            return None
        if name == "place_on_counter":
            if held is None:
                return "place_on_counter(): your hand is empty"
            if not self._empty_counters(world):
                return "place_on_counter(): no empty counter reachable from your side"
            return None
        if name == "chop":
            if held is None or hn in ("Plate",) or not hasattr(held, "chop_state"):
                return "chop(): you must hold an unchopped ingredient"
            if _is_chopped(held):
                return f"chop(): the {hn} you hold is already chopped"
            if not self._free_boards(world):
                return "chop(): no CutBoard reachable from your side (or all occupied)"
            return None
        if name == "put_on_plate":
            if held is None or not hasattr(held, "chop_state"):
                return "put_on_plate(): you must hold a chopped ingredient"
            if not _is_chopped(held):
                return f"put_on_plate(): the {hn} you hold is not chopped"
            p = _plate(world)
            if p is None:
                return "put_on_plate(): there is no Plate"
            if any(getattr(a, "holding", None) is p for a in world.agents):
                return "put_on_plate(): the Plate is being carried, not on a counter"
            if not stand_options(world, self.me, tuple(p.location)):
                return "put_on_plate(): the Plate is not reachable from your side"
            return None
        if name == "pickup_plate":
            if held is not None:
                return f"pickup_plate(): your hand is not empty (you hold {hn})"
            p = _plate(world)
            if p is None or any(getattr(a, "holding", None) is p for a in world.agents):
                return "pickup_plate(): the Plate is not on a counter"
            if not stand_options(world, self.me, tuple(p.location)):
                return "pickup_plate(): the Plate is not reachable from your side"
            return None
        if name == "deliver":
            if hn != "Plate":
                return "deliver(): you must hold the Plate"
            if not reachable_targets(world, self.me, "DeliverSquare"):
                return "deliver(): the DeliverSquare is not reachable from your side"
            return None
        if name == "wait":
            try:
                n = int(arg)
            except (TypeError, ValueError):
                return f"wait({arg}): n must be an integer in 1..{MAX_WAIT}"
            if not 1 <= n <= MAX_WAIT:
                return f"wait({n}): n must be in 1..{MAX_WAIT}"
            return None
        return f"{name}: unknown skill; allowed: {', '.join(SKILL_NAMES)}"

    def _empty_counters(self, world):
        occ = _occupied_cells(world)
        return [loc for loc in reachable_targets(world, self.me, "Counter") if loc not in occ]

    def _free_boards(self, world):
        occ = _occupied_cells(world)
        return [loc for loc in reachable_targets(world, self.me, "CutBoard") if loc not in occ]

    # ---- lifecycle ----
    def plan(self, name, arg):
        self.skill = (name, arg)
        self.steps = self.stuck = 0
        self.last_pos = None
        self._phase = "put" if name == "chop" else None
        self._board = self._item = None
        self.wait_left = int(arg) if name == "wait" else 0

    def done(self, world):
        name, arg = self.skill
        held = _held(world, self.me)
        hn = type(held).__name__ if held is not None else None
        if name == "pickup":
            return hn == arg
        if name == "place_on_counter":
            return held is None and self.steps > 0
        if name == "chop":
            return held is not None and _is_chopped(held)
        if name == "put_on_plate":
            p = _plate(world)
            return held is None and self._item is not None and any(c is self._item for c in getattr(p, "content", []))
        if name == "pickup_plate":
            return hn == "Plate"
        if name == "deliver":
            return held is None and self.steps > 0
        if name == "wait":
            return self.wait_left <= 0
        return True

    def _goal(self, world):
        """(target_cell) the current skill must act on now, or None if it cannot."""
        name, arg = self.skill
        held = _held(world, self.me)
        if name == "pickup":
            objs = [o for o in _loose_items(world) if type(o).__name__ == arg]
            cands = [tuple(o.location) for o in objs]
        elif name == "place_on_counter":
            cands = self._empty_counters(world)
        elif name == "chop":
            if self._phase == "put":
                if held is None:          # item already down (e.g. put completed last step)
                    self._phase = "chop"
                else:
                    self._item = held
                    if self._board is None or self._board in _occupied_cells(world):
                        fb = self._free_boards(world)
                        if not fb:
                            return None
                        self._board = min(fb, key=lambda l: len(stand_options(world, self.me, l)[0][2]))
                    return self._board
            if self._phase == "chop":
                if self._item is not None and _is_chopped(self._item):
                    self._phase = "take"
                else:
                    return self._board
            if self._phase == "take":
                return self._board
            return None
        elif name == "put_on_plate":
            self._item = held
            p = _plate(world)
            cands = [tuple(p.location)] if p is not None else []
        elif name == "pickup_plate":
            p = _plate(world)
            cands = [tuple(p.location)] if p is not None else []
        elif name == "deliver":
            cands = reachable_targets(world, self.me, "DeliverSquare")
        else:
            return None
        best = None
        for c in cands:
            opts = stand_options(world, self.me, c)
            if opts and (best is None or len(opts[0][2]) < best[0]):
                best = (len(opts[0][2]), c)
        return best[1] if best else None

    def step(self, world):
        """One env action for the current skill."""
        name, arg = self.skill
        self.steps += 1
        if name == "wait":
            self.wait_left -= 1
            return A.NOOP, ("done" if self.wait_left <= 0 else "running")
        fail = self.check_running(world)
        if fail:
            return A.NOOP, "failed:" + fail
        target = self._goal(world)
        if target is None:
            return A.NOOP, "failed:" + f"{skill_str(name, arg)}: target no longer available"
        me = world.agents[self.me]
        other = world.agents[1 - self.me]
        pos, ori = tuple(me.location), getattr(me, "orientation", None)
        # adjacent + facing -> interact
        if A.target_of(pos, ori) == target:
            if tuple(other.location) == target:
                return A.NOOP, "running"           # partner stands on the cell; wait
            act = A.INTERACT
        else:
            opts = stand_options(world, self.me, target, blocked=(tuple(other.location),)) \
                or stand_options(world, self.me, target)
            if not opts:
                return A.NOOP, "failed:" + f"{skill_str(name, arg)}: no path to the target"
            stand, facing, path = opts[0]
            if pos == stand:
                act = facing                       # turn in place (cell is not floor -> no move)
            else:
                nxt = path[1]
                act = _dir_to(pos, nxt)
                if nxt == tuple(other.location):
                    act = A.NOOP                   # blocked by the partner this step
        # unstick (ProAgent's auto_unstuck): no progress for STUCK_LIMIT steps -> random floor move
        if pos == self.last_pos and act != A.INTERACT:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pos = pos
        if self.stuck >= STUCK_LIMIT:
            floor = _floor(world)
            moves = [a for a in (A.LEFT, A.RIGHT, A.UP, A.DOWN)
                     if A.target_of(pos, a) in floor and A.target_of(pos, a) != tuple(other.location)]
            if moves:
                act = self.rng.choice(moves)
            self.stuck = 0
        return act, "running"

    def check_running(self, world):
        """Re-validate while running (ProAgent validates every step). Lenient on transient states."""
        name, arg = self.skill
        held = _held(world, self.me)
        if name == "pickup" and held is not None and type(held).__name__ != arg:
            return f"pickup({arg}): you are holding {type(held).__name__}"
        if name == "put_on_plate":
            p = _plate(world)
            if p is None or any(getattr(a, "holding", None) is p for a in world.agents):
                return "put_on_plate(): the Plate is being carried"
        if name == "deliver" and (held is None or type(held).__name__ != "Plate") and self.steps > 1:
            return "deliver(): you no longer hold the Plate"
        return None
