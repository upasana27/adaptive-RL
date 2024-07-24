import numpy as np
import itertools
from copy import deepcopy
from environment.mpe.core import World
from environment.policy_common import DynamicPolicy, MultiAgentResamplePolicy
from environment.mpe.env import MPE

action_deltas = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])


def get_near_action(src, dst):
    return np.sum(np.square(np.clip(src + 0.1 * action_deltas, -1.0, 1.0) - dst), axis=1).argmin()


def get_far_action(src, dst):
    return np.sum(np.square(np.clip(src + 0.1 * action_deltas, -1.0, 1.0) - dst), axis=1).argmax()


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def get_outbound_action(p):
    dim = np.abs(p).argmax()
    # print(p, dim)
    if p[dim] < -1:
        return dim * 2 + 1
    if p[dim] > 1:
        return dim * 2 + 2
    return 0


class PredatorPolicy:
    def __init__(self, num_predator, num_prey, pref):
        self.id = None
        self.prey_ids = list(range(num_predator, num_predator + num_prey))
        assert pref in ['N', 'S', 'E', 'W'] or pref in self.prey_ids
        # assert pref in ['N', 'S', 'E', 'W'] or pref in PreyPolicy.nav_landmarks
        self.pref = self.pref_ = pref

    def __str__(self):
        return f'PredatorPolicy({self.id}, {self.pref}, {self.pref_})'

    def __repr__(self):
        return str(self)

    # @staticmethod
    # def get_all_predator_policies(num_predator, num_prey):
    #     # print(f'Constructed {4 + num_prey} predator policies for {num_predator} predators and {num_prey} preys')
    #     # return [PredatorPolicy(num_predator, num_prey, pref) for pref in ['N', 'S', 'E', 'W']] + \
    #     #        [PredatorPolicy(num_predator, num_prey, pid) for pid in range(num_predator, num_predator + num_prey)]
    #     # return [PredatorPolicy(num_predator, num_prey, pid) for pid in range(num_predator, num_predator + num_prey)]
    #     return [PredatorPolicy(num_predator, num_prey, nav_name) for nav_name in PreyPolicy.nav_landmarks]
    @staticmethod
    def get_all_predator_policies(num_predator, num_prey):
        return [PredatorPolicy(num_predator, num_prey, prey_id) for prey_id in range(num_predator, num_predator + num_prey)]

    def set_id(self, aid):
        self.id = aid

    def reset(self):
        # Predator policies are stateless. No need to reset
        pass

    def __call__(self, world: World):
        # Get close to a specific prey
        agent = world.agents[self.id]
        self_pos = agent.state.p_pos
        ob_act = get_outbound_action(self_pos)
        if ob_act:
            # print(f'Predator {self.id} is out of bound at {self_pos}, action {ob_act}')
            return ob_act

        # visible_prey_ids = [pid for pid in self.prey_ids
        #                     if distance(self_pos, world.agents[pid].state.p_pos) <= world.obs_radius]
        # Fully observable predator teammate.
        visible_prey_ids = self.prey_ids.copy()

        if len(visible_prey_ids) == 0 or (self.pref in self.prey_ids and self.pref not in visible_prey_ids):
            # Can't see any prey or the preferred prey is not visible. Take a random action
            action = np.random.randint(5)
            # print(f'Predator {self.id} can\'t see any prey, action {action}')
            return action

        # prey_pos = [world.agents[pid].state.p_pos for pid in self.prey_ids]
        visible_prey_pos = [world.agents[pid].state.p_pos for pid in visible_prey_ids]
        if self.pref in self.prey_ids:
            # Prey with fixed index
            target_prey = visible_prey_ids.index(self.pref)
        elif self.pref == 'N':
            # Prey to a specific direction
            target_prey = np.argmin([pos[0] for pos in visible_prey_pos])
        elif self.pref == 'S':
            target_prey = np.argmax([pos[0] for pos in visible_prey_pos])
        elif self.pref == 'E':
            target_prey = np.argmax([pos[1] for pos in visible_prey_pos])
        elif self.pref == 'W':
            target_prey = np.argmin([pos[1] for pos in visible_prey_pos])
        else:
            raise ValueError('Invalid preference')
        action = get_near_action(self_pos, visible_prey_pos[target_prey])
        # print(f'Predator {self.id} is chasing prey {target_prey} at {prey_pos[target_prey]}, action {action}')
        return action
        # return get_near_action(self_pos, prey_pos[target_prey])


class PreyPolicy:
    # nav_landmarks = {
    #     'rectangle': [[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]],
    #     'circle': [[0.8 * np.cos(theta), 0.8 * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 8)[:-1]],
    #     'cross': [[-0.8, 0], [0.8, 0], [0, -0.8], [0, 0.8]],
    #     'triangle': [[0.8 * np.cos(theta), 0.8 * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 4)[:-1]],
    # }
    path_landmarks = [[0.8, 0], [0.8, 0.8], [0, 0.8], [-0.8, 0.8], [-0.8, 0], [-0.8, -0.8], [0, -0.8], [0.8, -0.8]]
    _path_landmarks = path_landmarks + path_landmarks
    nav_landmarks = {}
    for i in range(len(path_landmarks)):
        nav_landmarks[f'path_{i}'] = _path_landmarks[i:i + 3] + [_path_landmarks[i + 1]]

    def __init__(self, num_predator, pref, direction):
        self.id = None
        self.predator_ids = list(range(num_predator))
        assert pref in PreyPolicy.nav_landmarks or pref in self.predator_ids
        self.pref = pref
        # assert direction in [-1, 1]
        assert direction == 1
        self.dir = direction
        self.next_landmark = None

    def __str__(self):
        return f'PreyPolicy({self.id}, {self.pref}, {self.dir}, {self.next_landmark})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def get_all_prey_policies(num_predator):
        # print(f'Constructed {4 + num_predator} prey policies')
        # return [PreyPolicy(num_predator, pref) for pref in PreyPolicy.nav_landmarks] + \
        #        [PreyPolicy(num_predator, pid) for pid in range(num_predator)]
        # return [PreyPolicy(num_predator, pref, d * 2 - 1) for pref in PreyPolicy.nav_landmarks for d in range(2)]
        return [PreyPolicy(num_predator, pref, 1) for pref in PreyPolicy.nav_landmarks]

    @staticmethod
    def get_train_prey_policies(num_predator):
        return [PreyPolicy(num_predator, f'path_{i}', 1) for i in range(1, 8, 2)]

    @staticmethod
    def get_eval_prey_policies(num_predator):
        return [PreyPolicy(num_predator, f'path_{i}', 1) for i in range(0, 8, 2)]

    def reset(self):
        if self.pref in PreyPolicy.nav_landmarks:
            self.next_landmark = 0

    def set_id(self, aid):
        self.id = aid

    def __call__(self, world: World):
        agent = world.agents[self.id]
        self_pos = agent.state.p_pos
        ob_act = get_outbound_action(self_pos)
        if ob_act:
            # print(f'Prey {self.id} is out of bound, action {ob_act}')
            return ob_act

        if self.pref in PreyPolicy.nav_landmarks:
            # Go to the next landmark
            if np.linalg.norm(agent.state.p_pos - PreyPolicy.nav_landmarks[self.pref][self.next_landmark]) < 0.1:
                total_landmarks = len(PreyPolicy.nav_landmarks[self.pref])
                self.next_landmark = (self.next_landmark + self.dir + total_landmarks) % total_landmarks
            next_landmark_pos = PreyPolicy.nav_landmarks[self.pref][self.next_landmark]
            action = get_near_action(agent.state.p_pos, next_landmark_pos)
            # print(f'Prey {self.id} is going to landmark {self.next_landmark} at {next_landmark_pos}, action {action}')
            return action
            # return get_near_action(agent.state.p_pos, next_landmark_pos)
        elif self.pref in self.predator_ids:
            # Avoid a specific predator
            d = distance(agent.state.p_pos, world.agents[self.pref].state.p_pos)
            if d <= world.obs_radius:
                action = get_far_action(agent.state.p_pos, world.agents[self.pref].state.p_pos)
                # print(f'Prey {self.id} is avoiding predator {self.pref} at {world.agents[self.pref].state.p_pos}, action {action}')
            else:
                # Can't see the target predator. Take a random action
                action = np.random.randint(5)
            return action
            # return get_far_action(agent.state.p_pos, world.agents[self.pref].state.p_pos)
        else:
            raise ValueError('Invalid preference')


def get_all_policy_combinations(player_ids, num_predators, num_preys, all_predator_policies, all_prey_policies):
    # Returns a list of MultiAgentResamplePolicy.
    return [
        MultiAgentResamplePolicy(
            player_ids,
            policy_lists=None,
            current_policies=[deepcopy(all_predator_policies[i]) for i in predator_combinations]
                             + [deepcopy(all_prey_policies[i]) for i in prey_combinations],
            current_ids=predator_combinations + prey_combinations,
            max_ids=[len(all_predator_policies)] * (num_predators - 1) + [len(all_prey_policies)] * num_preys
        )
        for prey_combinations in itertools.permutations(range(len(all_prey_policies)), num_preys)
        for predator_combinations in itertools.permutations(range(len(all_predator_policies)), num_predators - 1)
    ]


def get_train_eval_pool(args):
    old_state = np.random.get_state()
    np.random.seed(args.pool_seed)

    # Generate all possible combinations
    num_preys = args.num_good_agents
    num_predators = args.num_agents - num_preys
    assert args.player_id < num_predators, f'Player id {args.player_id} is not one of the {num_predators} predators'
    predator_policies = PredatorPolicy.get_all_predator_policies(num_predators, num_preys)
    # assert len(predator_policies) == 4 + num_preys
    if args.separate_patterns:
        train_prey_policies = PreyPolicy.get_train_prey_policies(num_predators)
        eval_prey_policies = PreyPolicy.get_eval_prey_policies(num_predators)

        all_train_policies = get_all_policy_combinations([i for i in range(args.num_agents) if i != args.player_id],
                                                         num_predators, num_preys, predator_policies, train_prey_policies)
        all_train_policy_indices = np.arange(len(all_train_policies))
        print(f'{len(all_train_policies)} train combinations generated in total')
        assert len(all_train_policies) >= args.train_pool_size, \
            f'Not enough train combinations generated: {len(all_train_policies)} < {args.train_pool_size}'
        np.random.shuffle(all_train_policy_indices)

        all_eval_policies = get_all_policy_combinations([i for i in range(args.num_agents) if i != args.player_id],
                                                         num_predators, num_preys, predator_policies, eval_prey_policies)
        all_eval_policy_indices = np.arange(len(all_eval_policies))
        print(f'{len(all_eval_policies)} eval combinations generated in total')
        assert len(all_eval_policies) >= args.eval_pool_size, \
            f'Not enough eval combinations generated: {len(all_eval_policies)} < {args.eval_pool_size}'
        np.random.shuffle(all_eval_policy_indices)

        train_pool = [all_train_policies[i] for i in all_train_policy_indices[:args.train_pool_size]]
        eval_pool = [all_eval_policies[i] for i in all_eval_policy_indices[:args.eval_pool_size]]
    else:
        prey_policies = PreyPolicy.get_all_prey_policies(num_predators)
        # assert len(prey_policies) == 4 + num_predators

        all_policies = get_all_policy_combinations([i for i in range(args.num_agents) if i != args.player_id],
                                                   num_predators, num_preys, predator_policies, prey_policies)
        all_policy_indices = np.arange(len(all_policies))
        print(f'{len(all_policies)} combinations generated in total')
        assert len(all_policies) >= args.train_pool_size + args.eval_pool_size, \
            f'Not enough combinations generated: {len(all_policies)} < {args.train_pool_size} + {args.eval_pool_size}'
        np.random.shuffle(all_policy_indices)

        train_pool = [all_policies[i] for i in all_policy_indices[:args.train_pool_size]]
        eval_pool = [all_policies[i] for i in all_policy_indices[args.train_pool_size:args.train_pool_size + args.eval_pool_size]]

    for p in train_pool + eval_pool:
        # # Rename every predator's preference from a kind of policy to a specific prey ID
        # prey_policy_list = [p.current_policies[num_predators - 1 + i].pref for i in range(num_preys)]

        # Rename every predator's pref_ from a specific prey ID to a kind of policy
        for i in range(num_predators - 1):
            # assert prey_policy_list.count(p.current_policies[i].pref) == 1  # Sanity check
            # p.current_policies[i].pref = num_predators + prey_policy_list.index(p.current_policies[i].pref)
            assert isinstance(p.current_policies[p.current_policies[i].pref - 1], PreyPolicy)
            p.current_policies[i].pref_ = p.current_policies[p.current_policies[i].pref - 1].pref

    print(f'{len(train_pool)} training combinations and {len(eval_pool)} evaluation combinations generated:')
    for p in train_pool + eval_pool:
        print(p.current_policies)

    # # assert args.opponent_switch_period_min is None
    # if args.opponent_switch_period_min is not None:
    #     print('Switching opponents; make sure in eval mode')
    #     # print(eval_pool)
    #     # input()
    #     eval_pool_ = [DynamicPolicy([deepcopy(eval_pool[i]), deepcopy(eval_pool[(i + 1) % len(eval_pool)])],
    #                                 args.opponent_switch_period_min, args.opponent_switch_period_max)
    #                   for i in range(len(eval_pool))]
    #     eval_pool = eval_pool_
    #     for p in eval_pool:
    #         p.resample()

    np.random.set_state(old_state)

    return train_pool, eval_pool


if __name__ == '__main__':
    from argparse import Namespace
    arg = Namespace(
        num_agents=4,
        num_good_agents=2,
        horizon=100,
        player_id=0,
        history_size=10,
        train_pool_size=1,
        eval_pool_size=0,
        pool_seed=1,
        opponent_switch_period_min=None,
        opponent_switch_period_max=None,
        separate_patterns=False
    )
    np.random.seed(1230)
    env = MPE('simple_tag_multi_partial', arg.num_agents, arg.num_good_agents, arg.horizon, 0.2, 1.0,
              True, True, True, True)
    tp, ep = get_train_eval_pool(arg)
    print(tp, ep)
    env.set_id(arg.player_id)
    env.set_opponent(tp[0])
    obs = env.reset()
    print(obs)
    env.render(mode='human', close=False)
    while True:
        print('Current agent locations:', [agent.state.p_pos for agent in env.env.world.agents])
        print('Landmark locations:', [landmark.state.p_pos for landmark in env.env.world.landmarks])
        print('Agent types:', tp[0].current_policies)
        for pol in tp[0].current_policies[1:]:
            print(PreyPolicy.nav_landmarks[pol.pref], pol.next_landmark)
        while True:
            input_action = int(input('Input action...'))
            if input_action in range(5):
                break
        obs, _, done, info = env.step(input_action)
        print(obs, info)
        env.render(mode='human', close=False)
        if done:
            break
