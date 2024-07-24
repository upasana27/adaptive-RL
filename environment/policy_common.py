import numpy as np
from typing import List


class MultiAgentResamplePolicy:
    # Sample (possibly) multiple policies for multiple agents when reset
    def __init__(self, agent_ids, policy_lists, current_policies=None, current_ids=None, max_ids=None):
        if current_policies is not None:
            # Assigned manually and not resampled
            assert current_ids is not None and max_ids is not None
            assert len(current_policies) == len(current_ids) == len(max_ids) == len(agent_ids)
            for policy, pid in zip(current_policies, agent_ids):
                policy.set_id(pid)
            self.current_policies = current_policies
            self.current_ids = current_ids
            self.max_ids = max_ids
            self.policy_lists = None
        else:
            assert current_policies is None and current_ids is None and max_ids is None
            assert len(policy_lists) > 0 and all(len(policy_list) > 0 for policy_list in policy_lists)
            self.policy_lists = policy_lists
            assert len(policy_lists) == len(agent_ids)

            for policy_list, pid in zip(policy_lists, agent_ids):
                for pol in policy_list:
                    pol.set_id(pid)

            self.current_policies = None
            self.max_ids = [len(policy_list) for policy_list in policy_lists]
            self.current_ids = None

    def __str__(self):
        return f'MultiAgentResamplePolicy(policy_lists={self.policy_lists}, current_policies={self.current_policies}, current_ids={self.current_ids}, max_ids={self.max_ids})'

    def __repr__(self):
        return str(self)

    def resample(self):
        self.current_policies = [np.random.choice(policy_list) for policy_list in self.policy_lists]
        self.current_ids = [policy_list.index(policy)
                            for policy, policy_list in zip(self.current_policies, self.policy_lists)]
        # print('Resetting policies:', self.current_policies)

    def full_reset(self):
        self.reset()

    def reset(self):
        for pol in self.current_policies:
            pol.reset()

    def __call__(self, *args):
        # if len(self.policy_ids) == 1:
        #     return self.current_policies[0](obs_list)
        # return [pol(obs_list[pid]) for pol, pid in zip(self.current_policies, self.policy_ids)]
        if len(self.current_policies) == 1:
            return self.current_policies[0](*args)
        return [pol(*args) for pol in self.current_policies]
        # return [pol(args[0].agents[pid], *args) for pid, pol in zip(self.policy_ids, self.current_policies)]


class DynamicPolicy:
    def __init__(self, policies: List[MultiAgentResamplePolicy], period_min, period_max, period_schedule=None):
        self.policies = policies
        # print(policies)
        # input()
        self.period_min = period_min
        self.period_max = period_max
        if self.period_min is not None:
            assert 0 < self.period_min <= self.period_max
            assert period_schedule is None
        self.period_schedule = period_schedule
        if self.period_schedule is not None:
            assert len(self.period_schedule) == len(self.policies)
        self.policy = None
        # print(self.interval, self.period_min, self.period_max)
        self.period = self.period_idx = None
        self.max_ids = self.current_policies = None
        self.policy_idx = None
        # raise NotImplementedError('Check interaction reset code')

    def __str__(self):
        return f'DynamicPolicy(policy={self.policy}, policies={self.policies}, policy_idx={self.policy_idx}, period_min={self.period_min}, period_max={self.period_max}, period={self.period}, period_idx={self.period_idx})'

    def __repr__(self):
        return str(self)

    def full_reset(self):
        print('Full reset called')
        self.policy_idx = 0
        self.resample()

    def resample(self):
        # Resamples a period length
        if self.period_min is not None:
            self.period = np.random.randint(self.period_min, self.period_max + 1)
        else:
            self.period = self.period_schedule[self.policy_idx]
        self.period_idx = 0
        self.policy = self.policies[self.policy_idx]
        print('resample() called, using policy', self.policy_idx)
        # print('Using', self.policy_idx)
        self.policy_idx = (self.policy_idx + 1) % len(self.policies)
        if self.policy_idx == 0:
            print('Beware: policy overflowing')
        self.max_ids = self.policy.max_ids
        self.current_policies = self.policy.current_policies
        # self.policy.resample()

    def set_id(self, aid):
        for pol in self.policies:
            for p in pol.current_policies:
                p.set_id(aid)

    def reset(self):
        if self.period_idx == self.period:
            self.resample()
        self.period_idx += 1
        self.policy.reset()

    def __call__(self, *args):
        return self.policy(*args)
