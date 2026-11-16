"""LLMPolicy — an LLM acting as one agent in the Overcooked gridworld.

Interface-compatible with RuleBasedPolicy (set_id / reset / __call__(world) -> int action),
plus round memory: the driver calls start_round / start_episode /
observe_step / end_episode so the LLM sees what happened earlier in the round, mirroring
PACE's 5-episode history.

Two action modes:
  actions="low"  (default, unchanged): one LLM call per env step, reply = one of
                 left/right/up/down/interact/noop.
  actions="high": ProAgent-style skills (llm_agent/skills.py). The LLM is called only when a
                 skill completes or fails; a controller executes the low-level moves.

Fallback: model error or unparseable reply -> NOOP (low) / wait(1) (high), counted in client stats.
"""
from llm_agent import actions as A
from llm_agent.state_text import describe, describe_high, system_prompt, system_prompt_high
from llm_agent.memory import RoundMemory


def _ingredients_of(recipes):
    """'OnionLettuceSalad = Onion + Lettuce' -> {'Onion', 'Lettuce'}"""
    out = set()
    for r in recipes:
        if "=" in r:
            out.update(p.strip() for p in r.split("=", 1)[1].split("+"))
    return out


class LLMPolicy:
    def __init__(self, client, recipes, horizon, plan_len=1, verbose=False,
                 record=None, system_prompt_text=None, reason=False,
                 episodes_per_round=5, memory="full", actions="low"):
        self.client = client
        self.recipes = recipes
        self.horizon = horizon
        self.plan_len = max(1, int(plan_len))
        self.verbose = verbose
        self.record = record if record is not None else []
        self.E = episodes_per_round
        self.memory_mode = memory                 # "full" | "summary" | "none"
        self.actions_mode = actions               # "low" | "high"
        if actions == "high":
            self.system = system_prompt_text or system_prompt_high(self.E, horizon, memory != "none")
        else:
            self.system = system_prompt_text or system_prompt(self.E, horizon, memory != "none")
        self.reason = reason
        self.agent_id = 0
        self.mem = None
        self.ctrl = None
        self._queue, self._t, self._last_partner_action = [], 0, None
        self.last_prompt = self.last_reply = None
        self._reset_skill_state()

    def _reset_skill_state(self):
        self._skill_hist = []        # ["t3 pickup(Tomato): done", ...]
        self._beliefs = []           # [(t, predicted partner skill)]            (ProAgent intention)
        self._pbeliefs = []          # [(t, free-text belief about the partner)]  (
                                     #  open-ended, per episode, echoed back like the intentions)
        self._partner_real = []      # [(t, observed partner event)]
        self._feedback = None
        if self.ctrl is not None:
            self.ctrl.skill = None

    # ---- policy protocol --------------------------------------------------
    def set_id(self, aid):
        self.agent_id = aid
        self.mem = RoundMemory(aid, self.E, self.horizon)
        if self.actions_mode == "high":
            from llm_agent.skills import SkillController
            self.ctrl = SkillController(aid, _ingredients_of(self.recipes))

    def reset(self):
        """Per-EPISODE state only (step counter, cached plan, partner's last action)."""
        self._queue, self._t, self._last_partner_action = [], 0, None
        self._reset_skill_state()

    # ---- round memory -----------------------------------------------------
    def start_round(self, round_idx):
        self.mem.start_round(round_idx)

    def start_episode(self, ep_idx):
        self.reset()
        self.mem.start_episode(ep_idx)

    def observe_step(self, t, actions, events):
        self.mem.record(t, actions, events)
        if self.actions_mode == "high":
            for e in events.get(1 - self.agent_id, []):
                self._partner_real.append((t, e))

    def end_episode(self, success, t_end, plate_final):
        self.mem.end_episode(success, t_end, plate_final)

    def note_partner_action(self, a):
        self._last_partner_action = a

    def __call__(self, world):
        self._t += 1
        if self.actions_mode == "high":
            return self._call_high(world)
        if self._queue:
            return self._queue.pop(0)
        self.last_prompt = self.last_reply = None
        header = (f"ROUND {self.mem.round_idx + 1}, EPISODE {self.mem.ep_idx + 1} of {self.E}"
                  if self.mem is not None else None)
        mem_text = (self.mem.text(full_stack=(self.memory_mode == "full"))
                    if self.mem is not None and self.memory_mode != "none" else None)
        state = describe(world, self.agent_id, self.recipes, self.horizon,
                         self._t, self._last_partner_action, header=header, memory_text=mem_text)
        if self.reason:
            ask = ("\nThink in at most 3 short sentences, then end with a final line exactly of the form\n"
                   "ACTION: <left|right|up|down|interact|noop>")
        elif self.plan_len == 1:
            ask = "\nReply with exactly one action word."
        else:
            ask = (f"\nReply with up to {self.plan_len} action words separated by commas "
                   f"(a short plan). Stop the plan early if you expect the world to change.")
        try:
            text, reasoning = self.client.chat(self.system, state + ask)
        except Exception as e:
            self.record.append(dict(t=self._t, error=str(e)[:200], action=A.NOOP))
            self.last_prompt = state + ask
            return A.NOOP

        self.last_prompt, self.last_reply = state + ask, text
        if self.reason:
            a = A.parse_tagged(text)
            if a is None and not text and reasoning:
                a = A.parse_tagged(reasoning)
            acts = [a] if a is not None else []
        else:
            if not text and reasoning:
                text = reasoning if A.parse_tagged(reasoning) is not None else ""
                acts = [A.parse_tagged(text)] if text else []
            else:
                acts = [A.parse(p) for p in text.replace("\n", ",").split(",")]
            acts = [a for a in acts if a is not None][: self.plan_len]
        if not acts:
            self.client.n_unparseable += 1
            self.record.append(dict(t=self._t, raw=text[:200], action=A.NOOP, unparseable=True))
            return A.NOOP
        self.record.append(dict(t=self._t, raw=text[:200], actions=acts,
                                reasoning=(reasoning or "")[:300]))
        if self.verbose:
            print(f"[t={self._t}] {text[:60]!r} -> {[A.NAMES[a] for a in acts]}")
        self._queue = acts[1:]
        return acts[0]

    # ---- high-level (skill) mode -------------------------------------------
    def _belief_text(self):
        if not self._beliefs and not self._partner_real and not self._pbeliefs:
            return None
        pred = "; ".join(f"t{t}: {s}" for t, s in self._beliefs[-10:]) or "none yet"
        pb = "; ".join(f"t{t}: {s}" for t, s in self._pbeliefs[-10:]) or "none yet"
        real = "; ".join(f"t{t}: {e}" for t, e in self._partner_real[-10:]) or "nothing observed yet"
        return ("BELIEF (this episode)\n"
                f"  Your earlier predictions of the partner's intention: {pred}\n"
                f"  Your earlier beliefs about the partner: {pb}\n"
                f"  Partner's observed actions: {real}")

    @staticmethod
    def _parse_partner_belief(text):
        import re
        m = re.findall(r"^\s*\**Partner belief\**\s*:\s*(.*)$", text or "", flags=re.I | re.M)
        return m[-1].strip().strip('"') if m else None

    def _query_skill(self, world):
        """One LLM call -> (name, arg) or None if unparseable/error. Sets last_prompt/last_reply."""
        from llm_agent.skills import parse_skill, skill_str
        header = (f"ROUND {self.mem.round_idx + 1}, EPISODE {self.mem.ep_idx + 1} of {self.E}"
                  if self.mem is not None else None)
        mem_text = (self.mem.text(full_stack=(self.memory_mode == "full"))
                    if self.mem is not None and self.memory_mode != "none" else None)
        state = describe_high(world, self.agent_id, self.recipes, self.horizon, self._t,
                              header=header, memory_text=mem_text,
                              skill_history=self._skill_hist, belief_text=self._belief_text(),
                              feedback=self._feedback)
        self._feedback = None
        try:
            text, reasoning = self.client.chat(self.system, state)
        except Exception as e:
            self.record.append(dict(t=self._t, error=str(e)[:200]))
            self.last_prompt, self.last_reply = state, f"(error: {e})"
            return None
        intent = parse_skill(text, "Intention for partner")
        skill = parse_skill(text, "Plan")
        if skill is None and not text and reasoning:
            skill = parse_skill(reasoning, "Plan")
        self._beliefs.append((self._t, skill_str(*intent) if intent else "unknown"))
        self._pbeliefs.append((self._t, self._parse_partner_belief(text) or "unknown"))
        self.last_prompt = state
        self.last_reply = text + (f"\n[parsed plan: {skill_str(*skill)}]" if skill else "\n[parsed plan: NONE]")
        if skill is None:
            self.client.n_unparseable += 1
            self.record.append(dict(t=self._t, raw=text[:200], unparseable=True))
            return None
        self.record.append(dict(t=self._t, raw=text[:200], skill=skill_str(*skill),
                                reasoning=(reasoning or "")[:300]))
        if self.verbose:
            print(f"[t={self._t}] plan {skill_str(*skill)}")
        return skill

    def _call_high(self, world):
        from llm_agent.skills import MAX_REPLANS, skill_str
        ctrl = self.ctrl
        self.last_prompt = self.last_reply = None
        # finished skill? (checked against the world AFTER its last step)
        if ctrl.skill is not None and ctrl.done(world):
            self._skill_hist.append(f"t{self._t - 1} {skill_str(*ctrl.skill)}: done")
            ctrl.skill = None
        replans = 0
        while True:
            if ctrl.skill is None:
                skill = self._query_skill(world)
                if skill is None:
                    skill = ("wait", "1")
                    self._skill_hist.append(f"t{self._t} (unparseable reply -> wait(1))")
                fail = ctrl.check(world, *skill)
                if fail:
                    self._skill_hist.append(f"t{self._t} {skill_str(*skill)}: rejected ({fail})")
                    self._feedback = f"skill {skill_str(*skill)} failed its assertion: {fail}."
                    replans += 1
                    if replans > MAX_REPLANS:
                        ctrl.plan("wait", "1")
                    else:
                        continue
                else:
                    ctrl.plan(*skill)
            act, status = ctrl.step(world)
            if status.startswith("failed:"):
                reason = status[len("failed:"):]
                self._skill_hist.append(f"t{self._t} {skill_str(*ctrl.skill)}: failed ({reason})")
                self._feedback = f"skill {skill_str(*ctrl.skill)} failed: {reason}."
                ctrl.skill = None
                replans += 1
                if replans > MAX_REPLANS:
                    ctrl.plan("wait", "1")
                    act, _ = ctrl.step(world)
                    return act
                continue
            if status == "done":            # wait(n) finished on this step
                self._skill_hist.append(f"t{self._t} {skill_str(*ctrl.skill)}: done")
                ctrl.skill = None
            return act
