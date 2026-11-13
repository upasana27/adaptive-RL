"""Minimal Level-Based Foraging environment with typed fruits.

Design follows semitable/lb-foraging semantics:
  - 2D grid, two agents, food items at fixed cells.
  - Actions per agent: NOOP, UP, DOWN, LEFT, RIGHT, LOAD.
  - A food item is collected only if BOTH agents are orthogonally adjacent
    to it AND both issue LOAD on the same step (level-2 joint load).
  - Movement into a cell with food or another agent is blocked.

Added on top of standard LBF: each food has a discrete `type_id` in
[0, num_fruit_types). Agents see the type of each food in the observation.
This is the analogue of overcooked "recipes" / conventions.
"""

from __future__ import annotations

import numpy as np

NOOP, UP, DOWN, LEFT, RIGHT, LOAD = 0, 1, 2, 3, 4, 5
NUM_ACTIONS = 6

# (dy, dx) deltas for the four cardinal moves.
_MOVE_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


class Fruit:
    __slots__ = ('row', 'col', 'type_id', 'collected')

    def __init__(self, row: int, col: int, type_id: int):
        self.row = row
        self.col = col
        self.type_id = type_id
        self.collected = False


class LBFCore:
    """Fully-cooperative typed-fruit LBF.

    Reward is shared: +complete_reward per successful joint load,
    minus step_cost each step.
    """

    def __init__(
        self,
        grid_size: int = 8,
        num_agents: int = 2,
        num_fruit_types: int = 6,
        num_fruits_per_episode: int = 3,
        present_types: tuple | None = None,
        horizon: int = 40,
        complete_reward: float = 10.0,
        step_cost: float = 0.1,
        align_reward: float = 0.0,
        align_window: int = 5,
        seed: int = 0,
    ):
        assert num_agents == 2, "Cooperative LBF assumes 2 agents."
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_fruit_types = num_fruit_types
        self.num_fruits_per_episode = num_fruits_per_episode
        # Which fruit types are eligible to spawn this episode. If None, any.
        self.present_types = present_types
        self.horizon = horizon
        self.complete_reward = complete_reward
        self.step_cost = step_cost
        self.align_reward = align_reward
        self.align_window = align_window

        self.rng = np.random.RandomState(seed)

        self.agents: list[tuple[int, int]] = []
        self.fruits: list[Fruit] = []
        self.step_count = 0
        self._adj_step = [-10**6] * self.num_agents  # per-agent last step adjacent to a fruit

    # ------------------------------------------------------------------ setup
    def seed(self, sd: int) -> None:
        self.rng = np.random.RandomState(sd)

    def set_present_types(self, present_types) -> None:
        self.present_types = tuple(int(t) for t in present_types)

    def _empty_cells(self) -> list[tuple[int, int]]:
        occupied = set(self.agents)
        for f in self.fruits:
            occupied.add((f.row, f.col))
        return [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]

    def reset(self) -> list[np.ndarray]:
        self.agents = []
        self.fruits = []
        self.step_count = 0
        self._adj_step = [-10**6] * self.num_agents

        # Place agents on opposite halves of the grid so they don't start on top of each other.
        half = self.grid_size // 2
        left_cells = [(r, c) for r in range(self.grid_size) for c in range(half)]
        right_cells = [(r, c) for r in range(self.grid_size) for c in range(half, self.grid_size)]
        a0 = left_cells[self.rng.randint(len(left_cells))]
        a1 = right_cells[self.rng.randint(len(right_cells))]
        self.agents = [a0, a1]

        # Spawn fruits on free cells (not on agents, not on each other, with at least one free
        # adjacent cell so the agents can actually reach them).
        # When present_types is None, sample without replacement so each episode has
        # exactly one fruit of each type (up to num_fruits_per_episode).
        if self.present_types is None:
            all_types = list(range(self.num_fruit_types))
            self.rng.shuffle(all_types)
            type_sequence = all_types[:self.num_fruits_per_episode]
        else:
            # Explicit override (e.g. for legacy single-fruit experiments):
            # repeat the given types to fill num_fruits_per_episode slots.
            pt = list(self.present_types)
            type_sequence = [pt[i % len(pt)] for i in range(self.num_fruits_per_episode)]
        for type_id in type_sequence:
            free = self._empty_cells()
            self.rng.shuffle(free)
            placed = False
            for (r, c) in free:
                if self._has_free_neighbor(r, c, ignore_fruits=False):
                    self.fruits.append(Fruit(r, c, type_id))
                    placed = True
                    break
            if not placed and free:
                r, c = free[0]
                self.fruits.append(Fruit(r, c, type_id))

        return [self._observe(i) for i in range(self.num_agents)]

    # ----------------------------------------------------------------- helpers
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _cell_blocked(self, r: int, c: int, allow_agent: bool = False) -> bool:
        if not self._in_bounds(r, c):
            return True
        for f in self.fruits:
            if not f.collected and f.row == r and f.col == c:
                return True
        if not allow_agent:
            if (r, c) in self.agents:
                return True
        return False

    def _has_free_neighbor(self, r: int, c: int, ignore_fruits: bool = True) -> bool:
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc):
                continue
            if (nr, nc) in self.agents:
                continue
            if not ignore_fruits:
                blocked = False
                for f in self.fruits:
                    if not f.collected and f.row == nr and f.col == nc:
                        blocked = True
                        break
                if blocked:
                    continue
            return True
        return False

    def _adjacent_agents(self, r: int, c: int) -> list[int]:
        out = []
        for i, (ar, ac) in enumerate(self.agents):
            if abs(ar - r) + abs(ac - c) == 1:
                out.append(i)
        return out

    # --------------------------------------------------------------------- step
    def step(self, actions: list[int]):
        assert len(actions) == self.num_agents
        self.step_count += 1

        reward = -self.step_cost
        info: dict = {
            'loaded_fruit_type': None,
            'fruit_loaded': False,
            'agent_attempted_load': [False] * self.num_agents,
            'agent_adjacent_to_fruit': [False] * self.num_agents,
        }

        # 1) Resolve movement. Conflict policy: if two agents target the same
        # cell, neither moves. If an agent targets a blocked cell, it stays put.
        proposed = list(self.agents)
        for i, a in enumerate(actions):
            if a in _MOVE_DELTAS:
                dr, dc = _MOVE_DELTAS[a]
                nr, nc = self.agents[i][0] + dr, self.agents[i][1] + dc
                if self._cell_blocked(nr, nc, allow_agent=False):
                    proposed[i] = self.agents[i]
                else:
                    proposed[i] = (nr, nc)
        # head-on swap or same-target collisions: revert both
        if proposed[0] == proposed[1] and proposed[0] != self.agents[0]:
            proposed[0] = self.agents[0]
            proposed[1] = self.agents[1]
        if proposed[0] == self.agents[1] and proposed[1] == self.agents[0]:
            proposed[0] = self.agents[0]
            proposed[1] = self.agents[1]
        self.agents = proposed

        # 2) Track adjacency for alignment reward.
        for i in range(self.num_agents):
            ar, ac = self.agents[i]
            for f in self.fruits:
                if f.collected:
                    continue
                if abs(ar - f.row) + abs(ac - f.col) == 1:
                    info['agent_adjacent_to_fruit'][i] = True
                    self._adj_step[i] = self.step_count
                    break

        # 3) Resolve LOAD: any uncollected fruit with BOTH agents orthogonally
        # adjacent AND both issuing LOAD is collected.
        loaded_any = False
        loaded_type = None
        for i, a in enumerate(actions):
            if a == LOAD:
                info['agent_attempted_load'][i] = True
        if actions[0] == LOAD and actions[1] == LOAD:
            for f in self.fruits:
                if f.collected:
                    continue
                adj = self._adjacent_agents(f.row, f.col)
                if 0 in adj and 1 in adj:
                    f.collected = True
                    reward += self.complete_reward
                    loaded_any = True
                    loaded_type = f.type_id
                    break  # one joint load per step

        info['fruit_loaded'] = loaded_any
        info['loaded_fruit_type'] = loaded_type

        # 4) Optional alignment shaping (off by default).
        if self.align_reward > 0.0 and not loaded_any:
            if (
                self.step_count - self._adj_step[0] <= self.align_window
                and self.step_count - self._adj_step[1] <= self.align_window
                and info['agent_adjacent_to_fruit'][0]
                and info['agent_adjacent_to_fruit'][1]
            ):
                reward += self.align_reward

        # Terminate as soon as any joint load succeeds: the other fruits are distractors
        # whose purpose is to create the partner-identification challenge.
        done = self.step_count >= self.horizon or loaded_any
        if done:
            info['termination_info'] = 'completed' if loaded_any else 'timeout'
        obs = [self._observe(i) for i in range(self.num_agents)]
        return obs, reward, done, info

    # ----------------------------------------------------------- observations
    def observation_size(self) -> int:
        # Per-agent dense vector:
        #   self pos (row/grid, col/grid)                              -> 2
        #   other pos (row/grid, col/grid)                              -> 2
        #   for each fruit: row/grid, col/grid, present_flag, type-1hot -> 3 + num_fruit_types
        per_fruit = 3 + self.num_fruit_types
        return 2 + 2 + self.num_fruits_per_episode * per_fruit

    def _observe(self, agent_id: int) -> np.ndarray:
        g = float(self.grid_size)
        ar, ac = self.agents[agent_id]
        other_id = 1 - agent_id
        br, bc = self.agents[other_id]
        feats: list[float] = [ar / g, ac / g, br / g, bc / g]
        for idx in range(self.num_fruits_per_episode):
            if idx < len(self.fruits):
                f = self.fruits[idx]
                present = 0.0 if f.collected else 1.0
                feats.extend([f.row / g, f.col / g, present])
                onehot = [0.0] * self.num_fruit_types
                onehot[f.type_id] = present  # zero out the type signal once collected
                feats.extend(onehot)
            else:
                feats.extend([0.0, 0.0, 0.0])
                feats.extend([0.0] * self.num_fruit_types)
        return np.asarray(feats, dtype=np.float32)

    # --------------------------------------------------------------------- misc
    def render_ascii(self) -> str:
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for f in self.fruits:
            if not f.collected:
                grid[f.row][f.col] = str(f.type_id)
        for i, (r, c) in enumerate(self.agents):
            grid[r][c] = 'A' if i == 0 else 'B'
        return '\n'.join(''.join(row) for row in grid)
