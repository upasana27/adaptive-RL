"""Round memory for the LLM agent.

Mirrors PACE's history regime: a ROUND is `episodes_per_round` (=5) episodes with the same
partner; memory accumulates across the round and is cleared when a new round starts
(PACE: PeriodicHistoryStorage with clear_period = history_size = 5).

What is stored ("whole stack"):
  * every step of every episode in the round, as one compact line: both actions + the
    game events they caused (pick-ups, put-downs, chops, plating, delivery);
  * at the end of each episode, a deterministic SUMMARY built from those events
    (no LLM summarization, so it cannot hallucinate).
Events are computed by diffing the live world before and after each env step.
"""
from llm_agent.actions import NAMES

try:
    from gym_cooking.cooking_world.constants import ChopFoodStates
    CHOPPED = ChopFoodStates.CHOPPED
except Exception:                                  # pragma: no cover
    CHOPPED = None


def _name(o):
    return type(o).__name__ if o is not None else None


def _plate(world):
    ps = world.world_objects.get("Plate", [])
    return ps[0] if ps else None


def snapshot(world):
    """Everything needed to diff one step: holdings, plate contents, chop states, locations."""
    plate = _plate(world)
    foods = {}
    for cls, objs in world.world_objects.items():
        for o in objs:
            if hasattr(o, "chop_state"):
                foods[id(o)] = (o, getattr(o, "chop_state", None) == CHOPPED)
    return dict(
        hold=[a.holding for a in world.agents],
        plate=plate,
        plate_content=list(getattr(plate, "content", [])) if plate is not None else [],
        chopped={k: v[1] for k, v in foods.items()},
        foods={k: v[0] for k, v in foods.items()},
    )


def step_events(before, world, actions, delivered):
    """Return {agent_idx: [event strings]} for what each agent caused this step."""
    after = snapshot(world)
    ev = {0: [], 1: []}
    new_on_plate = [o for o in after["plate_content"] if all(o is not p for p in before["plate_content"])]
    for i in (0, 1):
        h0, h1 = before["hold"][i], after["hold"][i]
        if h0 is None and h1 is not None:
            if _name(h1) == "Plate":
                inner = [_name(c) for c in getattr(h1, "content", [])]
                ev[i].append(f"picked up the Plate {inner}" if inner else "picked up the empty Plate")
            else:
                ev[i].append(f"picked up {_name(h1)}" + (" (chopped)" if before["chopped"].get(id(h1)) else ""))
        elif h0 is not None and h1 is None:
            if _name(h0) == "Plate":
                if delivered:
                    ev[i].append("DELIVERED the Plate")
                else:
                    ev[i].append(f"put the Plate down at {tuple(h0.location)}")
            elif any(o is h0 for o in new_on_plate):
                ev[i].append(f"put {_name(h0)} on the Plate")
            else:
                ev[i].append(f"put {_name(h0)} down at {tuple(h0.location)}")
        elif h0 is not None and h1 is not None and h0 is not h1:
            ev[i].append(f"swapped {_name(h0)} for {_name(h1)}")
        elif _name(h0) == "Plate" and h1 is h0:
            for o in new_on_plate:
                ev[i].append(f"added {_name(o)} to the Plate it is holding")
    # chops: food that became chopped this step -> credit the agent that pressed interact
    for k, was in before["chopped"].items():
        if not was and after["chopped"].get(k):
            who = [i for i in (0, 1) if actions[i] == 5]
            for i in (who or [0, 1])[:1]:
                ev[i].append(f"chopped {_name(after['foods'][k])}")
    return ev


class RoundMemory:
    def __init__(self, me_id, episodes_per_round, horizon):
        self.me, self.E, self.H = me_id, episodes_per_round, horizon
        self.start_round(0)

    def start_round(self, round_idx):
        self.round_idx = round_idx
        self.past = []           # [{ep, lines, summary}]
        self.cur_lines = []
        self.cur_events = []     # (t, who, text)
        self.ep_idx = 0

    def start_episode(self, ep_idx):
        self.ep_idx = ep_idx
        self.cur_lines, self.cur_events = [], []

    def record(self, t, actions, events):
        me, them = self.me, 1 - self.me
        def part(i, who):
            e = "; ".join(events[i])
            return f"{who} {NAMES.get(actions[i], actions[i])}" + (f" -> {e}" if e else "")
        self.cur_lines.append(f"t{t}: {part(me, 'you')} | {part(them, 'partner')}")
        for i, who in ((me, "you"), (them, "partner")):
            for e in events[i]:
                self.cur_events.append((t, who, e))

    def end_episode(self, success, t_end, plate_final):
        def evs(who):
            xs = [f"{e} (t{t})" for t, w, e in self.cur_events if w == who]
            return "; ".join(xs) if xs else "nothing"
        outcome = (f"SUCCESS - delivered at step {t_end}" if success
                   else f"FAILED - no delivery within {self.H} steps")
        summary = (f"Episode {self.ep_idx + 1}: {outcome}. Plate at the end: {plate_final}.\n"
                   f"    Partner: {evs('partner')}.\n"
                   f"    You: {evs('you')}.")
        self.past.append(dict(ep=self.ep_idx, lines=list(self.cur_lines), summary=summary))

    def text(self, full_stack=True):
        L = [f"MEMORY (round {self.round_idx + 1}; same partner all round; cleared when the next round starts)"]
        if not self.past:
            L.append("  Earlier episodes this round: none (this is the first episode of the round).")
        else:
            n_succ = sum("SUCCESS" in p["summary"] for p in self.past)
            L.append(f"  Earlier episodes this round: {len(self.past)}, successful: {n_succ}.")
            for p in self.past:
                L.append("  " + p["summary"])
                if full_stack:
                    L.append("    Step log: " + " / ".join(p["lines"]))
        L.append(f"  Current episode {self.ep_idx + 1} so far: " +
                 (" / ".join(self.cur_lines) if self.cur_lines else "(just started)"))
        return "\n".join(L)
