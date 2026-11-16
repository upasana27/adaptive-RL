"""Action constants for the Overcooked gridworld.

Ground truth: environment/overcooked/gym_cooking/cooking_world/cooking_world.py:54
    AGENT_ACTIONS: 0: Noop, 1: Left, 2: Right, 3: Down, 4: Up, 5: Interact

Movement semantics (cooking_world.perform_agent_action):
  * actions 1-4 set the agent's orientation AND attempt to walk one cell.
  * action 5 interacts with the cell the agent currently FACES (its orientation),
    and is a no-op if the other agent is standing on that cell.
"""
NOOP, LEFT, RIGHT, DOWN, UP, INTERACT = 0, 1, 2, 3, 4, 5

NAMES = {NOOP: "noop", LEFT: "left", RIGHT: "right", DOWN: "down", UP: "up", INTERACT: "interact"}
BY_NAME = {v: k for k, v in NAMES.items()}

# (dx, dy) applied by a move/orientation. y grows DOWNWARD.
DELTA = {LEFT: (-1, 0), RIGHT: (1, 0), DOWN: (0, 1), UP: (0, -1), NOOP: (0, 0), INTERACT: (0, 0)}


def target_of(location, orientation):
    """Cell an agent at `location` with `orientation` would act on / move into."""
    dx, dy = DELTA.get(orientation, (0, 0))
    return (location[0] + dx, location[1] + dy)


def parse(text):
    """Parse a model action string into an int, or None if unparseable."""
    if text is None:
        return None
    t = str(text).strip().lower().strip('."\'')
    if t in BY_NAME:
        return BY_NAME[t]
    if t.isdigit() and int(t) in NAMES:
        return int(t)
    for name, idx in BY_NAME.items():           # tolerate "move left", "interact now"
        if name in t:
            return idx
    return None


def parse_tagged(text):
    """Parse the LAST 'ACTION: <word>' line. Used in reasoning mode, where free text
    contains direction words ("facing left") that the loose parser would grab."""
    if not text:
        return None
    import re
    m = re.findall(r"ACTION\s*:\s*\**\s*([A-Za-z]+)", text, flags=re.I)
    if not m:
        return None
    return BY_NAME.get(m[-1].lower())
