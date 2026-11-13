"""Rule-based L0 policies + L0 pool factory for fully-cooperative LBF.

Mirrors `environment/overcooked/policy.py` patterns:
  - `LBFRuleBasedPolicy(fruit_support_set, ...)` — analogue of
    `RuleBasedPolicy(ingredient_support_set=...)`. Specializes in a subset
    of fruit types; ignores all others.
  - `get_train_eval_pool(args)` — returns (train_policies, eval_policies)
    of L0 policies, mirroring the overcooked entry point used by `train_.py`.
  - `LBFPretrainedPolicy` — thin wrapper around overcooked's
    `PretrainedPolicy_test` with LBF-specific defaults (obs_shape=(10,),
    max_episode_length=80).
  - `get_train_eval_l1_pool_lbf` — builds an L1 partner pool from a spec
    string like "L1_FC_h80:1,2" for L2 training.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from environment.lbf.core import NOOP, UP, DOWN, LEFT, RIGHT, LOAD


# ---------------------------------------------------------------------------- L0
class LBFRuleBasedPolicy:
    """L0 specialist: forages only fruits whose type is in `fruit_support_set`.

    Behavior each step:
      1. Find the nearest uncollected supported fruit.
      2. If no such fruit exists, NOOP.
      3. If currently orthogonally adjacent to it, LOAD (joint with partner if
         partner also LOADs while adjacent).
      4. Otherwise, greedy Manhattan move toward an adjacent cell of the fruit.
         With probability `rand_p`, choose a random non-NOOP action instead
         (mirrors overcooked stochasticity injection).
    """

    def __init__(
        self,
        fruit_support_set,
        num_fruit_types: int,
        nav_p: float = 0.0,
        rand_p: float = 0.0,
        agent_id: int | None = None,
    ):
        self.fruit_support_set = tuple(int(f) for f in fruit_support_set)
        self.num_fruit_types = int(num_fruit_types)
        self.nav_p = float(nav_p)
        self.rand_p = float(rand_p)
        self.agent_id = agent_id
        # One-hot mask matching overcooked's `ingredient_support_set_id` convention
        # so downstream `set_desire` calls have a directly analogous handle.
        self.fruit_support_set_id = tuple(
            int(t in self.fruit_support_set) for t in range(self.num_fruit_types)
        )

    # ------------------------------------------------------------------ identity
    def set_id(self, aid: int) -> None:
        self.agent_id = aid

    def reset(self) -> None:
        return None

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_supported(self, world):
        my_pos = world.agents[self.agent_id]
        best_fruit = None
        best_dist = None
        for f in world.fruits:
            if f.collected:
                continue
            if f.type_id not in self.fruit_support_set:
                continue
            d = self._manhattan(my_pos, (f.row, f.col))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_fruit = f
        return best_fruit

    def _move_toward(self, world, target_rc):
        my_pos = world.agents[self.agent_id]
        dr = target_rc[0] - my_pos[0]
        dc = target_rc[1] - my_pos[1]
        # Mix vertical / horizontal preference with nav_p.
        prefer_horizontal = (abs(dc) > 0) and (
            abs(dc) >= abs(dr) or np.random.rand() < self.nav_p
        )
        candidates = []
        if prefer_horizontal and dc != 0:
            candidates.append(RIGHT if dc > 0 else LEFT)
            if dr != 0:
                candidates.append(DOWN if dr > 0 else UP)
        else:
            if dr != 0:
                candidates.append(DOWN if dr > 0 else UP)
            if dc != 0:
                candidates.append(RIGHT if dc > 0 else LEFT)
        for a in candidates:
            from environment.lbf.core import _MOVE_DELTAS

            d = _MOVE_DELTAS[a]
            nr = my_pos[0] + d[0]
            nc = my_pos[1] + d[1]
            # Allow moving toward the target even if blocked; the env will keep
            # us in place. Picking a still-legal cell helps in tight situations.
            if 0 <= nr < world.grid_size and 0 <= nc < world.grid_size:
                if (nr, nc) != world.agents[1 - self.agent_id]:
                    blocked = False
                    for fr in world.fruits:
                        if (
                            not fr.collected
                            and fr.row == nr
                            and fr.col == nc
                            and (nr, nc) != target_rc
                        ):
                            blocked = True
                            break
                    if not blocked:
                        return a
        # Fall back to first candidate or NOOP.
        return candidates[0] if candidates else NOOP

    # ---------------------------------------------------------------------- call
    def __call__(self, world):
        assert self.agent_id is not None, "set_id must be called before __call__"
        if np.random.rand() < self.rand_p:
            return int(np.random.randint(1, 6))  # any non-NOOP action

        fruit = self._nearest_supported(world)
        if fruit is None:
            return NOOP

        my_pos = world.agents[self.agent_id]
        d = self._manhattan(my_pos, (fruit.row, fruit.col))
        if d == 1:
            return LOAD
        return self._move_toward(world, (fruit.row, fruit.col))


def get_train_eval_pool(args):
    """L0 pool for L1 training.

    Single capability: 3 fruit types, 3 L0 specialists (one per fruit).
    The L1 trains against all 3 simultaneously and must learn to identify,
    within-episode, which fruit its current partner is committed to.

    Pool sizes come from `args.train_pool_size` / `args.eval_pool_size` and
    should equal `args.lbf_num_fruit_types` for the standard 3-fruit setup.
    """
    assert args.env_name == 'LBF'
    num_fruit_types = getattr(args, 'lbf_num_fruit_types', 3)

    train_pool, eval_pool = [], []
    for i in range(args.train_pool_size):
        fruit_id = i % num_fruit_types
        train_pool.append(
            LBFRuleBasedPolicy(
                fruit_support_set=(fruit_id,),
                num_fruit_types=num_fruit_types,
                nav_p=0.0,
                rand_p=0.0,
            )
        )
    for i in range(args.eval_pool_size):
        fruit_id = i % num_fruit_types
        eval_pool.append(
            LBFRuleBasedPolicy(
                fruit_support_set=(fruit_id,),
                num_fruit_types=num_fruit_types,
                nav_p=0.0,
                rand_p=0.0,
            )
        )

    print(f'[LBF] num_fruit_types={num_fruit_types}')
    print(
        f'[LBF] train pool: {len(train_pool)} L0 specialists, '
        f'support sets = {[p.fruit_support_set for p in train_pool]}'
    )
    print(
        f'[LBF] eval pool: {len(eval_pool)} L0 specialists, '
        f'support sets = {[p.fruit_support_set for p in eval_pool]}'
    )
    return train_pool, eval_pool


# ---------------------------------------------------------------------------- L1 pretrained (for L2 training)

def _make_lbf_pretrained_policy(args, checkpoint_path, horizon):
    """Instantiate a PretrainedPolicy_test configured for LBF."""
    from environment.overcooked.policy import PretrainedPolicy_test
    num_fruits = getattr(args, 'lbf_num_fruits_per_episode', 3)
    num_types = getattr(args, 'lbf_num_fruit_types', 3)
    # obs: 2 (self pos) + 2 (other pos) + num_fruits * (2 pos + 1 present + num_types one-hot)
    obs_shape = (4 + num_fruits * (3 + num_types),)
    return PretrainedPolicy_test(
        latent_training=args.latent_training,
        history_size=args.history_size,
        has_rew_done=args.has_rew_done,
        has_meta_time_step=args.has_meta_time_step,
        include_current_episode=args.include_current_episode,
        merge_encoder_computation=args.merge_encoder_computation,
        last_episode_only=args.last_episode_only,
        pop_oldest_episode=args.pop_oldest_episode,
        self_obs_mode=args.self_obs_mode,
        model_path=checkpoint_path,
        eval_history=True,
        agent_id=1 - args.player_id,
        obs_shape=obs_shape,
        max_episode_length=horizon,
    )


def get_train_eval_l1_pool_lbf(args, train_spec, eval_spec, horizon=80):
    """Build train + eval pools of pretrained L1 agents for LBF L2 training.

    Spec format: "L1_FC_h80:1,2,3" or "L1_FC_h80:1,2;L1_FC_h80:3"
    Checkpoint path: logs/LBF/ppo_june_new_exp/{TYPE}/seed{N}_seed{N}/ppo/latest.pt
    """
    repo_root = os.environ.get('PACE_ROOT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
    BASE = os.path.join(repo_root, 'logs', 'LBF', 'ppo_june_new_exp')

    def load_pool(spec, label):
        pool = []
        for type_spec in spec.split(';'):
            agent_type, seeds_str = type_spec.strip().split(':')
            seeds = [int(s.strip()) for s in seeds_str.split(',')]
            for seed in seeds:
                ckpt = f'{BASE}/{agent_type}/seed{seed}_seed{seed}/ppo/latest.pt'
                if not os.path.exists(ckpt):
                    raise FileNotFoundError(f'[LBF L1 pool] {label} checkpoint not found: {ckpt}')
                pool.append(_make_lbf_pretrained_policy(args, ckpt, horizon))
                print(f'  [LBF L1 {label}] loaded {agent_type} seed {seed}')
        return pool

    print('=' * 60)
    print(f'[LBF] Loading L1 pool for L2 training (horizon={horizon})')
    print(f'  player_id={args.player_id}, agent_id for partners={1 - args.player_id}')
    train_pool = load_pool(train_spec, 'train')
    eval_pool = load_pool(eval_spec, 'eval')
    print(f'[LBF] train={len(train_pool)} partners, eval={len(eval_pool)} partners')
    print('=' * 60)
    return train_pool, eval_pool
