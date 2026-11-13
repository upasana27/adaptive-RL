"""Serialize the Overcooked world into text an LLM can act on.

Reads the live CookingWorld (env.get_env_world()) rather than the dense obs vector,
because the 73/115-dim observation is not interpretable by a language model.
"""
from llm_agent.actions import NAMES, target_of

try:
    from gym_cooking.cooking_world.constants import ChopFoodStates
    CHOPPED = ChopFoodStates.CHOPPED
except Exception:                                  # pragma: no cover
    CHOPPED = None

# Object classes that are scenery, not ingredients/containers.
_STATIC = ("Counter", "Floor", "CutBoard", "DeliverSquare", "Blender", "Plate")


def _chop_str(obj):
    st = getattr(obj, "chop_state", None)
    if st is None:
        return ""
    return " (CHOPPED)" if CHOPPED is not None and st == CHOPPED else " (unchopped)"


def _held_str(agent):
    h = getattr(agent, "holding", None)
    if h is None:
        return "nothing"
    name = type(h).__name__
    if name == "Plate":
        inner = [type(c).__name__ for c in getattr(h, "content", [])]
        return f"Plate containing {inner}" if inner else "an empty Plate"
    return f"{name}{_chop_str(h)}"


def _on_plate(world, obj):
    """True if obj is an ingredient already placed on a Plate (it moves with the plate)."""
    return any(obj is c for p in world.world_objects.get("Plate", []) for c in getattr(p, "content", []))


def _static_at(world, loc):
    """Name of the static tile at loc ('Counter'/'Floor'/'CutBoard'/'DeliverSquare')."""
    for cls_name in ("CutBoard", "DeliverSquare", "Blender", "Counter", "Floor"):
        objs = world.world_objects.get(cls_name, [])
        for o in objs:
            if tuple(o.location) == tuple(loc):
                return cls_name
    return "unknown"


# ---------------------------------------------------------------------------------
# MAP: the level layout as defined in gym_cooking/utils/new_style_level/<level>.json
# ("-" = counter, floor shown as "." instead of a space so it can be counted), with the
# live positions of objects and agents drawn on top. Object positions are randomized per
# episode within ranges in that json, so the map is rendered from the live world.
# ---------------------------------------------------------------------------------
def _legend_items(world, me_id):
    me, other = world.agents[me_id], world.agents[1 - me_id]
    marks, legend = {}, []
    for o in world.world_objects.get("CutBoard", []):
        marks[tuple(o.location)] = "X"
    for o in world.world_objects.get("DeliverSquare", []):
        marks[tuple(o.location)] = "D"
    for p in world.world_objects.get("Plate", []):
        if not any(getattr(a, "holding", None) is p for a in world.agents):
            marks[tuple(p.location)] = "P"
    n = 0
    for cls_name, objs in sorted(world.world_objects.items()):
        if cls_name in _STATIC:
            continue
        for o in objs:
            if not hasattr(o, "location") or _on_plate(world, o):
                continue
            if any(getattr(a, "holding", None) is o for a in world.agents):
                continue
            n += 1
            ch = chr(ord("a") + n - 1)        # single character keeps the grid aligned (FI has 12)
            marks[tuple(o.location)] = ch
            legend.append(f"{ch}={cls_name}")
    marks[tuple(other.location)] = "B"
    marks[tuple(me.location)] = "A"
    return marks, legend


def render_map(world, me_id):
    floor = {tuple(o.location) for o in world.world_objects.get("Floor", [])}
    marks, legend = _legend_items(world, me_id)
    rows = ["    x: " + " ".join(str(x) for x in range(world.width))]
    for y in range(world.height):
        cells = []
        for x in range(world.width):
            c = marks.get((x, y))
            cells.append(c if c else ("." if (x, y) in floor else "-"))
        rows.append(f"  y={y:<2} " + " ".join(cells))
    key = ("Legend: - counter (not walkable)   . floor   A = YOU   B = your partner   "
           "X = CutBoard   D = DeliverSquare   P = Plate (on a counter)   "
           + "   ".join(legend))
    return "\n".join(rows) + "\n" + key


def sides(world, me_id):
    """Facts derived from the layout: which floor region each agent stands in, what each
    can reach, and the counter cells shared by both regions (the divider)."""
    me, other = world.agents[me_id], world.agents[1 - me_id]
    mine, theirs = set(_reachable(world, me.location)), set(_reachable(world, other.location))
    def touch(region, loc):
        x, y = loc
        return any((x + dx, y + dy) in region for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    xs = sorted({x for x, _ in mine})
    side = "LEFT" if xs and xs[-1] < world.width / 2 else "RIGHT"
    L = [f"YOUR SIDE: you are on the {side} side (you can walk on floor x={xs[0]}..{xs[-1]}). "
         f"Your partner is on the other side. Neither of you can cross the divider."]
    for d in world.world_objects.get("DeliverSquare", []):
        loc = tuple(d.location)
        if touch(mine, loc) and not touch(theirs, loc):
            L.append(f"DELIVERY: the DeliverSquare {loc} is on YOUR side. You can deliver; your partner CANNOT.")
        elif touch(theirs, loc) and not touch(mine, loc):
            L.append(f"DELIVERY: the DeliverSquare {loc} is on your PARTNER's side. You CANNOT reach it, "
                     f"so you can never deliver; only your partner can.")
        else:
            L.append(f"DELIVERY: the DeliverSquare {loc} is reachable by both of you.")
    shared = sorted(tuple(c.location) for c in world.world_objects.get("Counter", [])
                    if touch(mine, tuple(c.location)) and touch(theirs, tuple(c.location)))
    shared += sorted(tuple(o.location) for k in ("CutBoard",) for o in world.world_objects.get(k, [])
                     if touch(mine, tuple(o.location)) and touch(theirs, tuple(o.location)))
    if shared:
        occupied = {tuple(o.location) for k, objs in world.world_objects.items() if k not in _STATIC
                    for o in objs if hasattr(o, "location")} | \
                   {tuple(p.location) for p in world.world_objects.get("Plate", [])}
        free = [c for c in shared if c not in occupied]
        L.append(f"DIVIDER: counter cells both of you can reach (the only way to pass items): {shared}. "
                 f"Currently empty (you can put an item down there): {free}.")
    return "\n".join(L)


def describe(world, me_id, recipes, horizon, t, last_partner_action=None, header=None,
             memory_text=None):
    """Return a compact text description of the world from agent `me_id`'s view."""
    me = world.agents[me_id]
    other = world.agents[1 - me_id]
    L = []
    L.append((f"{header}, " if header else "") + f"STEP {t} of {horizon}  (you are agent {me_id})")

    # --- layout (same grid as the level json) + side facts ---
    L.append(f"MAP ({world.width} wide x {world.height} tall; x grows RIGHT, y grows DOWN):")
    L.append(render_map(world, me_id))
    # sides(world, me_id) (YOUR SIDE / DELIVERY / DIVIDER lines) removed from the prompt 2026-10-19 per user.

    # --- me ---
    face = target_of(me.location, getattr(me, "orientation", 1))
    L.append(f"YOU: at {tuple(me.location)}, facing {NAMES.get(getattr(me,'orientation',1),'?')} "
             f"-> cell {face} which is a {_static_at(world, face)}. You hold: {_held_str(me)}.")
    # --- partner ---
    pa = f", last action: {NAMES.get(last_partner_action, last_partner_action)}" if last_partner_action is not None else ""
    L.append(f"PARTNER: at {tuple(other.location)}. Holds: {_held_str(other)}{pa}.")

    # --- plate(s) ---
    for p in world.world_objects.get("Plate", []):
        inner = [type(c).__name__ + _chop_str(c) for c in getattr(p, "content", [])]
        what = 'contains ' + str(inner) if inner else 'EMPTY'
        if getattr(me, "holding", None) is p:
            L.append(f"PLATE: held by YOU, {what}")
        elif getattr(other, "holding", None) is p:
            L.append(f"PLATE: held by your PARTNER, {what}")
        else:
            L.append(f"PLATE at {tuple(p.location)}: {what}")

    # --- ingredients ---
    ing_lines = []
    for cls_name, objs in sorted(world.world_objects.items()):
        if cls_name in _STATIC:
            continue
        for o in objs:
            if not hasattr(o, "location"):
                continue
            # skip items currently held or already on a plate (reported above)
            if any(getattr(a, "holding", None) is o for a in world.agents) or _on_plate(world, o):
                continue
            ing_lines.append(f"  {cls_name} at {tuple(o.location)}{_chop_str(o)}")
    if ing_lines:
        L.append("INGREDIENTS on the map:")
        L.extend(sorted(ing_lines))

    # --- stations ---
    for cls_name in ("CutBoard", "DeliverSquare"):
        locs = [tuple(o.location) for o in world.world_objects.get(cls_name, [])]
        if locs:
            L.append(f"{cls_name}(s) at {locs}")

    L.append(f"RECIPES that score (need BOTH ingredients chopped and plated, then delivered):")
    for r in recipes:
        L.append(f"  {r}")

    # AFFORDANCES: for each item on MY side, where to stand and which way to face.
    # This is adjacency INFORMATION, not a solution - the model still has to choose the
    # recipe, sequence the subtasks, and route itself there.
    aff = affordances(world, me_id)
    if aff:
        L.append("WHAT YOU CAN REACH AND ACT ON (stand on the cell, face that way, then interact); "
                 "anything not listed is on your partner's side and out of your reach:")
        L.extend(aff)
    else:
        L.append("(no reachable interactable found on your side)")
    if memory_text:
        L.append(memory_text)
    return "\n".join(L)


def _reachable(world, start):
    """BFS over floor tiles from `start`. Returns {cell: steps}. The partner is ignored
    as an obstacle (it moves), so distances are a lower bound when it is in the way."""
    from collections import deque
    floor = {tuple(o.location) for o in world.world_objects.get("Floor", [])}
    start = tuple(start)
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n in floor and n not in dist:
                dist[n] = dist[(x, y)] + 1
                q.append(n)
    return dist


def affordances(world, me_id, max_items=14):
    """For each interactable I can actually REACH: the cell to stand on + facing needed.

    Fixed 2026-10-19: the old version listed every object on the map (including the
    partner's side, behind the divider) with Manhattan distances, and for divider items
    (e.g. the Plate) it could pick a stand cell on the PARTNER's side. Now only stand
    cells in my BFS-reachable floor region are used, and distance is true walking steps.
    """
    from llm_agent.actions import LEFT, RIGHT, UP, DOWN
    me = world.agents[me_id]
    reach = _reachable(world, me.location)
    out = []
    for cls_name, objs in world.world_objects.items():
        if cls_name in ("Counter", "Floor"):
            continue
        for o in objs:
            if not hasattr(o, "location"):
                continue
            if any(getattr(a, "holding", None) is o for a in world.agents) or _on_plate(world, o):
                continue
            loc = tuple(o.location)
            best = None
            for (dx, dy), word in (((-1, 0), "left"), ((1, 0), "right"),
                                   ((0, -1), "up"), ((0, 1), "down")):
                stand = (loc[0] - dx, loc[1] - dy)
                if stand in reach and (best is None or reach[stand] < best[0]):
                    best = (reach[stand], stand, word)
            if best is None:
                continue                      # not reachable from my side
            d, stand, word = best
            extra = _chop_str(o)
            if cls_name == "Plate":
                inner = [type(c).__name__ for c in getattr(o, "content", [])]
                extra = f" (contains {inner})" if inner else " (empty)"
            out.append((d, f"  {cls_name} at {loc}{extra}: stand at {stand}, face {word} "
                           f"({d} steps from you)"))
    out.sort()
    return [t for _, t in out[:max_items]]


_SYSTEM_BASE = """You are an expert player of a two-agent gridworld cooking game. You control ONE agent; the other agent is your partner and you cannot control or message it.

GOAL
Together you must complete one recipe: both of its ingredients chopped, placed on the single shared Plate, and the Plate delivered to the DeliverSquare. You score only when a full valid recipe is delivered. Only an agent whose side touches the DeliverSquare can deliver; the score is shared.

THE MAP
- A vertical wall of counters splits the map. You can only reach ingredients on YOUR side. Your partner reaches the other side's ingredients. The divider counter is reachable from both sides.
- Walls and counters are NOT walkable. You walk on floor tiles only.

ACTIONS (one per turn)
- left, right, up, down : turn to face that direction, then step one cell that way IF that cell is floor.
                          If that cell is a counter, wall, or holds an item, you DO NOT move: you only turn to face it.
                          This is how you face an ingredient on a counter: stand next to it and press its direction.
- interact             : act on the single cell you are currently FACING.
- noop                 : do nothing.

HOW INTERACT WORKS (this is the only way to affect anything)
- Facing a counter with an item on it, hands empty                  -> you PICK IT UP.
- Facing a CutBoard that holds an unchopped item, hands empty       -> you CHOP it.
- Facing an empty counter/CutBoard while holding an item           -> you PUT IT DOWN there.
- Facing the Plate while holding a chopped ingredient              -> you ADD it to the Plate.
- Facing the DeliverSquare while holding the completed Plate       -> you DELIVER.
- interact does NOTHING if your partner is standing on the cell you face.

CRITICAL RULES
- To act on something you must be ADJACENT to it and FACING it. Move so the item is exactly one cell in your facing direction, then interact.
- Chopping and plating are IRREVERSIBLE. There is only ONE Plate. If a wrong ingredient goes on the Plate, the recipe can never be completed.
- Picking up and putting down are reversible."""

_ROUNDS = """ROUNDS AND EPISODES
- You play in ROUNDS. One round = {E} episodes in a row with the SAME partner.
- Every episode starts from a fresh kitchen: agents, ingredients and the Plate are put back at starting positions (positions can differ between episodes). Nothing in the kitchen carries over between episodes.
- An episode SUCCEEDS if a valid recipe is delivered within {H} steps; the episode then ends immediately. Otherwise it ends after step {H} as a failure.
- Your ROUND SCORE is the number of successful episodes out of {E}. Every episode counts equally."""

_MEMORY = """
- MEMORY: each turn you are shown, for every earlier episode of this round, a summary (outcome, what your partner and you did) and its step-by-step log, plus the log of the current episode so far. Memory is cleared when a new round starts."""


def system_prompt(episodes_per_round=5, horizon=40, has_memory=True):
    """System message. The ROUNDS section explains the 5-episode round regime (PACE's
    meta-episode); the MEMORY line is included only when round memory is shown."""
    s = _SYSTEM_BASE + "\n\n" + _ROUNDS.format(E=episodes_per_round, H=horizon)
    if has_memory:
        s += _MEMORY
    return s


SYSTEM_PROMPT = system_prompt()



# ---------------------------------------------------------------------------------
# HIGH-LEVEL (skill) mode, ProAgent-style — added 2026-10-20. See llm_agent/skills.py.
# Same game rules as the low-level prompt; the ACTIONS/INTERACT sections are replaced by
# the skill set with its preconditions written as pseudo-code (ProAgent's convention), and
# the reply format is ProAgent's Analysis / Intention / Plan. No strategy demos (hints).
# ---------------------------------------------------------------------------------
_SKILLS = """SKILLS (you choose ONE skill per decision; a controller then walks there and performs the
low-level moves for you. You are asked again when the skill is DONE or FAILS. Every env step
still counts against the episode limit.)
def pickup(X):            # X = an ingredient name, e.g. pickup(Tomato)
    assert hand_empty()
    assert X is lying on a counter on YOUR side (not held, not already on the Plate)
def chop():               # chop the ingredient you hold on a CutBoard on your side; you end up holding it CHOPPED
    assert you hold an UNCHOPPED ingredient
    assert a CutBoard on your side is free
def put_on_plate():       # add the ingredient you hold to the Plate (IRREVERSIBLE)
    assert you hold a CHOPPED ingredient
    assert the Plate is on a counter reachable from your side
def place_on_counter():   # put down what you hold on the nearest empty counter on your side (reversible)
    assert you hold something
def pickup_plate():       # pick the Plate up from its counter
    assert hand_empty()
    assert the Plate is on a counter reachable from your side
def deliver():            # carry the Plate to the DeliverSquare and deliver it
    assert you hold the Plate
    assert the DeliverSquare is reachable from your side
def wait(n):              # do nothing for n env steps (1 <= n <= 10)
If a skill's assertion fails, you are told which one and asked to plan again."""

_FORMAT_HIGH = """REPLY FORMAT (exactly these four lines, nothing after them):
Analysis: <at most 3 short sentences about the scene and what your partner is doing>
Intention for partner: <the skill you think your partner is doing now, e.g. pickup(Onion), or unknown>
Partner belief: <what you believe about this partner from what you have seen so far, in your own words, or unknown>
Plan: <ONE skill call from the list, e.g. pickup(Tomato)>"""


def system_prompt_high(episodes_per_round=5, horizon=40, has_memory=True):
    base = _SYSTEM_BASE
    a = base.index("ACTIONS (one per turn)")
    b = base.index("CRITICAL RULES")
    rules = base[:a] + _SKILLS + "\n\n" + base[b:]
    # drop the low-level "ADJACENT and FACING" rule: the controller handles it
    rules = "\n".join(l for l in rules.splitlines() if "ADJACENT" not in l)
    s = rules + "\n\n" + _ROUNDS.format(E=episodes_per_round, H=horizon)
    if has_memory:
        s += "\n" + _MEMORY
    return s + "\n\n" + _FORMAT_HIGH


def describe_high(world, me_id, recipes, horizon, t, header=None, memory_text=None,
                  skill_history=None, belief_text=None, feedback=None):
    """State text for skill mode: map + state (coordinates only), recipes, this episode's skill
    history, belief block (ProAgent), round memory, and the last failure feedback if any.
    2026-10-20 (user decision): no '(side: ...)' tags — which side a thing is on is left for the
    model to read off the map; a wrong guess is answered by the controller's assertion feedback."""
    me = world.agents[me_id]
    other = world.agents[1 - me_id]
    L = []
    L.append((f"{header}, " if header else "") + f"STEP {t} of {horizon}  (you are agent {me_id})")
    L.append(f"MAP ({world.width} wide x {world.height} tall; x grows RIGHT, y grows DOWN):")
    L.append(render_map(world, me_id))
    L.append(f"YOU: at {tuple(me.location)}. You hold: {_held_str(me)}.")
    L.append(f"PARTNER: at {tuple(other.location)}. Holds: {_held_str(other)}.")
    for p in world.world_objects.get("Plate", []):
        inner = [type(c).__name__ + _chop_str(c) for c in getattr(p, "content", [])]
        what = 'contains ' + str(inner) if inner else 'EMPTY'
        if getattr(me, "holding", None) is p:
            L.append(f"PLATE: held by YOU, {what}")
        elif getattr(other, "holding", None) is p:
            L.append(f"PLATE: held by your PARTNER, {what}")
        else:
            L.append(f"PLATE at {tuple(p.location)}: {what}")
    ing = []
    for cls_name, objs in sorted(world.world_objects.items()):
        if cls_name in _STATIC:
            continue
        for o in objs:
            if not hasattr(o, "location"):
                continue
            if any(getattr(a, "holding", None) is o for a in world.agents) or _on_plate(world, o):
                continue
            ing.append(f"  {cls_name} at {tuple(o.location)}{_chop_str(o)}")
    if ing:
        L.append("INGREDIENTS on the map:")
        L.extend(sorted(ing))
    for cls_name in ("CutBoard", "DeliverSquare"):
        locs = [tuple(o.location) for o in world.world_objects.get(cls_name, [])]
        if locs:
            L.append(f"{cls_name}(s) at {locs}")
    L.append("RECIPES that score (need BOTH ingredients chopped and plated, then delivered):")
    for r in recipes:
        L.append(f"  {r}")
    if skill_history:
        L.append("YOUR SKILLS THIS EPISODE so far: " + " / ".join(skill_history))
    if belief_text:
        L.append(belief_text)
    if memory_text:
        L.append(memory_text)
    if feedback:
        L.append(f"CONTROLLER: {feedback} Plan again.")
    return "\n".join(L)
