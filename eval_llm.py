#!/usr/bin/env python
"""Evaluate an LLM as one agent in Overcooked, against L0 (rule-based) or L1 (neural) partners.

Credentials:  export LLM_BASE_URL and LLM_API_KEY   (any OpenAI-compatible endpoint)

Examples
--------
# LLM (LEFT seat) vs the FI R1R2 rule-based L0 pool, 3 meta-episodes of 5, horizon 80
python eval_llm.py --model gemma4-31b-it --domain FI --l0-pair-type R1R2 \
    --env-config environment/overcooked/config/fc_small_6recipe_h80.yaml \
    --player-id 0 --test-pass 3 --test-episodes 5 --out-dir logs/Overcooked/llm_eval/demo

# LLM vs an L1 neural partner (see WARNING below about the FI partner path)
python eval_llm.py --model gemma4-31b-it --domain FI \
    --partner-ckpt logs/.../L1_FI_pace_p0/right/R1R2/seed7/ppo/30000000.pt ...

Protocol: --test-pass rounds x --test-episodes episodes. The training regime is a
5-episode meta-episode (history clears every 5), so --test-episodes 5 matches it.

WARNING: neural L1 partners must be sanity-checked. Print the partner's action entropy
(reported at the end): near log(6)=1.792 means it is acting at random and the run is
meaningless (this exact bug bit the FI partner path in online_test_.py on 2026-10-17).
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment/overcooked"))

from environment.overcooked.overcooked_maker import OvercookedMaker
from llm_agent import actions as A
from llm_agent.client import ChatClient
from llm_agent.policy import LLMPolicy
from gym_cooking import Arrdict

COMPLETE_REWARD = 10.0


def build_partner(args, obs_shape, horizon=40):
    """Return (partner, kind) where kind is 'L0' (takes world) or 'L1' (takes obs)."""
    if args.partner_ckpt:
        from environment.overcooked.policy import PretrainedPolicy_test
        p = PretrainedPolicy_test(
            latent_training=True, history_size=args.history_size, has_rew_done=False,
            has_meta_time_step=False, include_current_episode=True,
            merge_encoder_computation=True, last_episode_only=False,
            pop_oldest_episode=False, self_obs_mode=True,
            model_path=args.partner_ckpt, eval_history=True,
            agent_id=1 - args.player_id, obs_shape=obs_shape,
            max_episode_length=horizon)   # history buffer must hold a full episode (H=80 for FI h80)
        return p, "L1"
    if args.domain == "FI":
        from environment.overcooked.policy import generate_policy_pool_6recipe
        pool = generate_policy_pool_6recipe(
            [args.l0_pair_type] if len(args.l0_pair_type) == 2 else
            {"R1R2": ["R1", "R2"], "R3R4": ["R3", "R4"], "R5R6": ["R5", "R6"]}[args.l0_pair_type],
            p_max=args.p, env_name="fc_small_divider_6recipe")
        return pool[args.l0_index % len(pool)], "L0"
    from environment.overcooked.policy import RuleBasedPolicy
    pairs = {"R1": ("Lettuce", "Onion"), "R2": ("Potato", "Broccoli"), "R3": ("Tomato", "Carrot")}
    l, r = pairs[args.l0_pair_type if args.l0_pair_type in pairs else "R1"]
    return RuleBasedPolicy("full", args.p, args.p, args.p, None, "fc_small_divider_test",
                           ingredient_support_set=[l, r]), "L0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gateway model id, e.g. gemma4-31b-it")
    ap.add_argument("--env-config", default="environment/overcooked/config/fc_small_6recipe_h80.yaml")
    ap.add_argument("--domain", choices=["FI", "FC"], default="FI")
    ap.add_argument("--l0-pair-type", default="R1R2")
    ap.add_argument("--l0-index", type=int, default=0, help="which L0 of the pair to face")
    ap.add_argument("--partner-ckpt", default=None, help="neural L1 partner instead of an L0")
    ap.add_argument("--player-id", type=int, default=0, choices=[0, 1])
    ap.add_argument("--p", type=float, default=0.0, help="L0 stochasticity")
    ap.add_argument("--test-pass", type=int, default=3)
    ap.add_argument("--test-episodes", type=int, default=5)
    ap.add_argument("--plan-len", type=int, default=1, help=">1 asks for a short action plan (cheaper)")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="reasoning models spend 2000+ tokens on a 1500-token state prompt; "
                         "too low and they return EMPTY content (finish_reason=length)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--history-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--memory", choices=["full", "summary", "none"], default="full",
                    help="round memory shown to the LLM: full = per-episode summary + full step log of "
                         "every episode this round; summary = summaries of earlier episodes + current "
                         "episode log; none = no memory")
    ap.add_argument("--no-think", action="store_true",
                    help="disable native thinking (Qwen3: chat_template_kwargs.enable_thinking=False)")
    ap.add_argument("--reason", action="store_true",
                    help="ask for <=3 sentences of visible reasoning then 'ACTION: <word>'")
    ap.add_argument("--trace-file", default=None,
                    help="write every step's exact prompt, reply, actions and resulting state")
    ap.add_argument("--actions", choices=["low", "high"], default="low",
                    help="low (default): one LLM call per env step, reply = a move. "
                         "high: ProAgent-style skills (pickup/chop/put_on_plate/place_on_counter/"
                         "pickup_plate/deliver/wait) executed by a controller; the LLM is called "
                         "only when a skill completes or fails (llm_agent/skills.py, 2026-10-20).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = OvercookedMaker.make_env_from_config(args.env_config)
    horizon = env.horizon
    with open(args.env_config) as f:
        import yaml
        raw = [r.replace("Salad", "") for r in yaml.safe_load(f)["recipes"]]
    # The config lists each recipe twice (forward + reverse ingredient order), which
    # reads as 12 distinct recipes to an LLM. Collapse to canonical unordered pairs and
    # name the two ingredients explicitly.
    import re as _re
    def _split(name):
        parts = _re.findall(r"[A-Z][a-z]+", name)
        return parts if len(parts) == 2 else [name]
    canon = {}
    for r in raw:
        parts = _split(r)
        canon[tuple(sorted(parts))] = f"{''.join(parts)}Salad = {parts[0]} + {parts[1]}" \
            if len(parts) == 2 else r
    recipes = sorted(canon.values())
    obs_shape = env.get_observation_space().obs.shape

    client = ChatClient(args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                        timeout=args.timeout, log_path=os.path.join(args.out_dir, "llm_calls.jsonl"),
                        default_extra_body=({"chat_template_kwargs": {"enable_thinking": False}}
                                            if args.no_think else None))
    llm = LLMPolicy(client, recipes, horizon, plan_len=args.plan_len, verbose=args.verbose,
                    reason=args.reason, episodes_per_round=args.test_episodes, memory=args.memory,
                    actions=args.actions)
    from llm_agent.memory import snapshot as _snap, step_events as _events
    trace = open(args.trace_file, "w") if args.trace_file else None
    def _wstate(w):
        out = []
        for i, ag in enumerate(w.agents):
            h = getattr(ag, "holding", None)
            out.append(f"agent{i} at {tuple(ag.location)} facing "
                       f"{A.NAMES.get(getattr(ag, 'orientation', -1))} holding "
                       f"{type(h).__name__ if h else None}")
        return "; ".join(out)
    partner, kind = build_partner(args, obs_shape, horizon)
    me, them = args.player_id, 1 - args.player_id
    llm.set_id(me)
    if hasattr(partner, "set_id"):
        partner.set_id(them)

    print(f"env={args.env_config} horizon={horizon} obs={obs_shape} recipes={recipes}")
    print(f"LLM model={args.model} seat=player_{me} ({'LEFT' if me==0 else 'RIGHT'})  "
          f"partner={kind} ({args.partner_ckpt or args.l0_pair_type+'['+str(args.l0_index)+'] p='+str(args.p)})")
    print(f"protocol: {args.test_pass} passes x {args.test_episodes} episodes, plan_len={args.plan_len}, "
          f"actions={args.actions}\n")

    llm_actions, partner_actions, episodes = Counter(), Counter(), []
    t_start = time.time()
    if kind == "L1" and args.test_episodes != args.history_size:
        raise SystemExit(f"--test-episodes ({args.test_episodes}) must equal --history-size "
                         f"({args.history_size}): the L1 history clears every history_size episodes "
                         f"and the partner is reset every round; they must line up (PACE: 5).")
    for pass_i in range(args.test_pass):
        # ROUND start (= PACE meta-episode): L1 partner reloaded with an empty history,
        # LLM round memory cleared.
        if hasattr(partner, "reset"):
            partner.reset()
        llm.start_round(pass_i)
        for ep_i in range(args.test_episodes):
            llm.start_episode(ep_i)   # per-episode: step counter, cached plan, last partner action
            env.seed(args.seed * 1000 + pass_i * 100 + ep_i)
            data = env.reset()
            obs = {k: v.obs for k, v in data.items()}
            total, delivered, t_deliver = 0.0, False, None
            first = {"llm": None, "partner": None}
            for t in range(horizon):
                world = env.get_env_world()
                a_me = llm(world)
                o_them = obs[f"player_{them}"]
                a_them = partner(world) if kind == "L0" else partner(o_them)
                a_them = int(np.asarray(a_them).flatten()[0])
                if trace:
                    trace.write(f"\n===== pass {pass_i} ep {ep_i} t={t+1}\n")
                    if llm.last_prompt is not None:
                        trace.write(f"--- PROMPT ---\n{llm.last_prompt}\n--- REPLY ---\n{llm.last_reply}\n")
                    else:
                        trace.write("(executing cached plan, no call)\n")
                    trace.write(f"--- ACTIONS: llm={A.NAMES[a_me]} partner={A.NAMES[a_them]}\n")
                llm.note_partner_action(a_them)
                llm_actions[a_me] += 1
                partner_actions[a_them] += 1
                if a_me == A.INTERACT and first["llm"] is None:
                    first["llm"] = t + 1
                if a_them == A.INTERACT and first["partner"] is None:
                    first["partner"] = t + 1
                before = _snap(world)
                # Insert in PLAYER-INDEX order: cooking_zoo applies actions by dict insertion order,
                # not by key, so inserting player_{me} first swapped the two agents' actions whenever
                # the LLM sat in seat 1 (found 2026-11-04: all seat-1 FC runs scored 0.0).
                acts_by_id = {me: a_me, them: a_them}
                dec = Arrdict()
                for pid in (0, 1):
                    dec[f"player_{pid}"] = Arrdict(action=acts_by_id[pid])
                data, _info = env.step(dec)
                obs = {k: v.obs for k, v in data.items()}
                if kind == "L1":
                    # Feed the L1 partner its history exactly as training does (learning/storage_.py
                    # insert(), self_obs_mode): (its own obs_t, the OTHER agent's action_t) each step,
                    # then the terminal obs + finish_episode. PeriodicHistoryStorage clears it every
                    # history_size (=5) episodes, and partner.reset() runs once per pass (= 5 episodes).
                    # Nothing in the codebase did this before 2026-10-19: without it the FC L1 pool scores
                    # 0.00 with its own L0 training partners; with it 0.8-1.0.
                    import torch as _t
                    partner.history.add(0, _t.as_tensor(o_them).float().cuda(), _t.tensor(a_me).cuda(), None)
                    if bool(data[f"player_{me}"].done):
                        partner.history.add(0, _t.as_tensor(obs[f"player_{them}"]).float().cuda(), None, None)
                        partner.history.finish_episode(0)
                r = float(data[f"player_{me}"].reward)
                total += r
                if trace:
                    trace.write(f"--- AFTER: reward={r:+.2f} | {_wstate(env.get_env_world())}\n")
                    trace.flush()
                if r >= COMPLETE_REWARD * 0.9 and not delivered:
                    delivered, t_deliver = True, t + 1
                acts_pair = {me: a_me, them: a_them}
                llm.observe_step(t + 1, acts_pair,
                                 _events(before, env.get_env_world(), acts_pair, delivered and t_deliver == t + 1))
                if bool(data[f"player_{me}"].done):
                    break
            _pl = env.get_env_world().world_objects.get("Plate", [])
            _plate_final = [type(c).__name__ for c in getattr(_pl[0], "content", [])] if _pl else []
            llm.end_episode(delivered, t_deliver if delivered else t + 1, _plate_final)
            episodes.append(dict(pass_i=pass_i, ep_i=ep_i, reward=total, success=int(delivered),
                                 t_deliver=t_deliver, first_llm=first["llm"],
                                 first_partner=first["partner"]))
            print(f"  pass {pass_i} ep {ep_i}: success={int(delivered)} reward={total:+.2f} "
                  f"first_interact llm={first['llm']} partner={first['partner']}  "
                  f"[{client.n_calls} calls, {client.stats()['sec_per_call']}s/call]")

    def ent(c):
        n = sum(c.values())
        if not n:
            return float("nan")
        p = np.array([c[k] / n for k in range(6)])
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    succ = [e["success"] for e in episodes]
    summary = dict(
        model=args.model, seat=f"player_{me}", partner=kind,
        partner_spec=args.partner_ckpt or f"{args.l0_pair_type}[{args.l0_index}] p={args.p}",
        env_config=args.env_config, horizon=horizon,
        passes=args.test_pass, episodes_per_pass=args.test_episodes, n_episodes=len(episodes),
        success_rate=float(np.mean(succ)), mean_reward=float(np.mean([e["reward"] for e in episodes])),
        llm_action_entropy=round(ent(llm_actions), 3),
        partner_action_entropy=round(ent(partner_actions), 3),
        uniform_entropy=round(float(np.log(6)), 3),
        llm_action_dist={A.NAMES[k]: llm_actions[k] for k in range(6)},
        wall_seconds=round(time.time() - t_start, 1), llm=client.stats())
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(dict(summary=summary, episodes=episodes), f, indent=2)

    print("\n" + "=" * 70)
    print(f"SUCCESS RATE: {summary['success_rate']:.3f}   mean reward {summary['mean_reward']:+.2f}"
          f"   ({len(episodes)} episodes)")
    print(f"LLM action entropy {summary['llm_action_entropy']} "
          f"(uniform={summary['uniform_entropy']}) dist={summary['llm_action_dist']}")
    print(f"partner action entropy {summary['partner_action_entropy']}")
    print(f"LLM calls: {client.stats()}")
    if summary["llm_action_entropy"] > 1.70:
        print("!! WARNING: LLM action distribution is near-uniform — check prompts/parsing "
              "before treating this as a capability result.")
    if client.n_unparseable or client.n_fail:
        print(f"!! {client.n_unparseable} unparseable + {client.n_fail} failed calls fell back to NOOP")
    print(f"wrote {os.path.join(args.out_dir, 'summary.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
