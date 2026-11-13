"""LBF env smoke test.

Two sub-tests:

  TEST 1: joint-load mechanics.
    Two L0s with the SAME fruit_support_set={0}. Env spawns a single fruit of
    type 0. Both L0s home in on the same fruit and joint-load it. Expected:
    all 5 episodes terminate with `completed`, reward > 0.

  TEST 2: capability disjointness.
    One L0 with fruit_support_set={0}, one with fruit_support_set={2}. Env
    spawns fruits of types 0 and 2. Each L0 only goes to "its" fruit; they
    never meet, never joint-load. Expected: 0% success rate.
    (This mirrors L1-vs-L1 across-type pairing, which V4 in the plan expects
    to be ~0% success.)

Usage:
    python -m scripts.lbf.smoke_test
"""

from __future__ import annotations

import numpy as np

from environment.lbf.core import LBFCore
from environment.lbf.policy import LBFRuleBasedPolicy


def run_episode(env, p0, p1) -> dict:
    env.reset()
    p0.set_id(0)
    p1.set_id(1)
    total_reward = 0.0
    loads = 0
    info = {}
    for step in range(env.horizon):
        a0 = p0(env)
        a1 = p1(env)
        _, r, done, info = env.step([a0, a1])
        total_reward += r
        if info['fruit_loaded']:
            loads += 1
        if done:
            return {'reward': total_reward, 'loads': loads,
                    'steps': step + 1,
                    'termination': info.get('termination_info')}
    return {'reward': total_reward, 'loads': loads, 'steps': env.horizon,
            'termination': info.get('termination_info', 'incomplete')}


def test_joint_load(num_episodes: int = 5) -> None:
    print("TEST 1: joint-load mechanics (matched L0s, single fruit type)")
    rewards, terms, loads = [], [], []
    for seed in range(num_episodes):
        np.random.seed(seed)
        env = LBFCore(
            grid_size=8, num_agents=2, num_fruit_types=3,
            num_fruits_per_episode=1, present_types=(0,),
            horizon=40, complete_reward=10.0, step_cost=0.1, seed=seed,
        )
        p0 = LBFRuleBasedPolicy(fruit_support_set=(0,), num_fruit_types=3)
        p1 = LBFRuleBasedPolicy(fruit_support_set=(0,), num_fruit_types=3)
        r = run_episode(env, p0, p1)
        rewards.append(r['reward'])
        terms.append(r['termination'])
        loads.append(r['loads'])
        print(f"  ep {seed}: reward={r['reward']:.2f} loads={r['loads']} "
              f"steps={r['steps']} term={r['termination']}")
    mean_reward = float(np.mean(rewards))
    success = float(np.mean([t == 'completed' for t in terms]))
    print(f"  -> mean reward={mean_reward:.2f}, success={success * 100:.0f}%")
    assert success > 0.6, f"Expected joint load to work; got success={success}"
    print("  PASSED.\n")


def test_disjoint_capability(num_episodes: int = 5) -> None:
    print("TEST 2: disjoint capability (L0s on different fruit types)")
    success_count = 0
    for seed in range(num_episodes):
        np.random.seed(seed)
        env = LBFCore(
            grid_size=8, num_agents=2, num_fruit_types=3,
            num_fruits_per_episode=2, present_types=(0, 2),
            horizon=40, complete_reward=10.0, step_cost=0.1, seed=seed,
        )
        p0 = LBFRuleBasedPolicy(fruit_support_set=(0,), num_fruit_types=3)
        p1 = LBFRuleBasedPolicy(fruit_support_set=(2,), num_fruit_types=3)
        r = run_episode(env, p0, p1)
        if r['termination'] == 'completed':
            success_count += 1
        print(f"  ep {seed}: reward={r['reward']:.2f} loads={r['loads']} "
              f"steps={r['steps']} term={r['termination']}")
    print(f"  -> success={success_count}/{num_episodes}")
    assert success_count == 0, "Disjoint L0s should never joint-load."
    print("  PASSED.\n")


def main() -> None:
    print("LBF env smoke test\n")
    test_joint_load()
    test_disjoint_capability()
    print("All tests PASSED.")


if __name__ == '__main__':
    main()
