import torch
from learning.utils import AgentIndicesMapper
# from learning.model import LatentPolicy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _to_chunk_batch(tensor: torch.Tensor, batch_size: int, chunk_size: int):
    # [num_steps, num_agents, *] -> [num_agents * num_chunks, chunk_size, *]
    return tensor.transpose(0, 1).reshape(batch_size, chunk_size, *tensor.size()[2:])


def _to_rnn_input(tensor: torch.Tensor, device):
    # [mini_batch_size, chunk_size, *] -> [chunk_size * mini_batch_size, *]
    return tensor.transpose(0, 1).reshape(-1, *tensor.size()[2:]).to(device)


class PeriodicHistoryStorage:
    # Periodic history storage
    # Maintains several periods of histories; each is stored as a list of episodes
    # Hold a partial trajectory for every process, several complete trajectories for every policy
    # Partial trajectories are stored as separate time steps in a list
    # Complete trajectories are stored as a unified tensor
    # Assume that process k maps to policy k % num_policies
    def __init__(self, num_processes, num_policies, history_storage_size, clear_period, refresh_interval, sample_size,
                 has_rew_done, max_samples_per_period, step_mode, use_episodes,
                 has_meta_time_step, include_current_episode, obs_shape, act_shape, max_episode_length,
                 merge_encoder_computation, last_episode_only, pop_oldest_episode):
        print(f'Building periodic history storage for {num_processes} processes '
              f'with size per process {history_storage_size}, total size {num_processes * history_storage_size}, '
              f'refresh interval {refresh_interval}, clear period {clear_period} (potentially +1 episode)')
        self.device = None
        assert use_episodes
        self.use_episodes = use_episodes
        self.max_size_per_process = history_storage_size
        self.clear_period = clear_period
        print('Using', 'episodes' if self.use_episodes else 'steps', 'as the unit of size in history buffer')
        assert self.max_size_per_process >= self.clear_period, \
            f'History storage size per process {self.max_size_per_process} must be >= clear period {self.clear_period}'
        self.refresh_interval = refresh_interval
        assert self.clear_period >= self.refresh_interval, \
            f'Clear period {self.clear_period} must be >= refresh interval {refresh_interval}'
        self.sample_size = sample_size
        self.num_opponents = num_policies
        self.num_processes = num_processes
        self.has_rew_done = has_rew_done
        self.has_meta_time_step = has_meta_time_step
        if self.has_meta_time_step:
            assert self.use_episodes
        assert not step_mode, 'Step mode is not supported'
        self.step_mode = step_mode
        self.include_current_episode = include_current_episode
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        # +1 here for the terminal observation.
        self.max_episode_length = max_episode_length + 1
        self.max_samples_per_period = max_samples_per_period
        self.merge_encoder_computation = merge_encoder_computation
        self.last_episode_only = last_episode_only
        self.pop_oldest_episode = pop_oldest_episode
        # Always store on CPU
        self.storage_device = 'cpu'

        # history_obs, history_act: tuple of histories
        # history: list of periods, not exceeding history_storage_size steps in total
        # period: tensor of (clear_period, max_episode_length, obs_shape or act_shape)

        # history_sizes: tuple of history sizes
        # history_size: list of period sizes
        # period_size: tensor of episode sizes
        # episode_size: scalar
        # history_total_size: list of scalars, each scalar is the total size of a history

        # If include_current_episode is set, then the current episode will be included in the history_obs tensor
        # current_* will also be disabled
        self.history_total_size = [0 for _ in range(num_processes)]
        self.history_sizes = tuple([] for _ in range(num_processes))
        self.history_obs = tuple([] for _ in range(num_processes))
        self.history_act = tuple([] for _ in range(num_processes))
        self.history_step_mask = tuple([] for _ in range(num_processes))
        self.current_episode = [0 for _ in range(num_processes)]
        self.current_step = [0 for _ in range(num_processes)]
        for i in range(num_processes):
            self.start_new_period(i)

        if merge_encoder_computation:
            self.history_episode_cache = tuple([] for _ in range(num_processes))
        else:
            self.history_episode_cache = None

    def initialize_cache(self):
        for i in range(self.num_processes):
            self.history_episode_cache[i].clear()
            self.history_episode_cache[i].extend([None] * len(self.history_sizes[i]))

    def start_new_period(self, proc_idx):
        self.history_sizes[proc_idx].append(
            torch.zeros((self.clear_period,), dtype=torch.long, device=self.storage_device)
        )
        # No steps in the new period by default
        self.history_step_mask[proc_idx].append(
            torch.ones((self.clear_period, self.max_episode_length), dtype=torch.bool, device=self.storage_device)
        )
        self.history_obs[proc_idx].append(
            torch.zeros((self.clear_period, self.max_episode_length) + self.obs_shape, device=self.storage_device)
        )
        self.history_act[proc_idx].append(
            torch.zeros((self.clear_period, self.max_episode_length) + self.act_shape,
                        dtype=torch.long, device=self.storage_device)
        )
        self.current_episode[proc_idx] = 0
        self.current_step[proc_idx] = 0

    def to(self, device):
        self.device = device

    def add(self, proc_idx, obs, act, rew=None):
        # assert not torch.isnan(obs).any()

        # This was initially after the has_rew_done check
        # Moved here to guarantee that the last element of an observation is always the done mark
        if self.has_meta_time_step:
            meta_step_tensor = torch.tensor([len(self.history_sizes[proc_idx][-1]) / self.clear_period]).to(obs.device)
            obs = torch.cat([obs, meta_step_tensor])

        if self.has_rew_done:
            assert rew is not None
            # Default done is 0. Will be changed to 1 in finish_episode
            done_tensor = torch.zeros(1).to(obs.device)
            rew_tensor = rew.unsqueeze(0).to(obs.device) / 10.0
            obs = torch.cat([obs, rew_tensor, done_tensor], dim=-1)

        if act is None:
            # Action doesn't exist. This -1 will be converted into an all-zero vector
            act = torch.tensor(-1).to(self.storage_device)

        self.history_obs[proc_idx][-1][self.current_episode[proc_idx], self.current_step[proc_idx]] = obs
        self.history_act[proc_idx][-1][self.current_episode[proc_idx], self.current_step[proc_idx]] = act
        self.history_step_mask[proc_idx][-1][self.current_episode[proc_idx], self.current_step[proc_idx]] = False
        self.history_sizes[proc_idx][-1][self.current_episode[proc_idx]] += 1
        self.history_total_size[proc_idx] += 1
        self.current_step[proc_idx] += 1

    def pop_episode(self, proc_idx):
        # Pop the oldest one
        assert len(self.history_obs[proc_idx]) == 1
        self.current_episode[proc_idx] -= 1

        # Handle sizes
        self.history_total_size[proc_idx] -= self.history_sizes[proc_idx][0][0].item()
        self.history_sizes[proc_idx][0][:-1] = self.history_sizes[proc_idx][0][1:].clone()
        self.history_sizes[proc_idx][0][-1] = 0

        # Move data forward
        self.history_obs[proc_idx][0][:-1] = self.history_obs[proc_idx][0][1:].clone()
        self.history_act[proc_idx][0][:-1] = self.history_act[proc_idx][0][1:].clone()
        self.history_step_mask[proc_idx][0][:-1] = self.history_step_mask[proc_idx][0][1:].clone()

        # Clear masks
        self.history_step_mask[proc_idx][0][-1] = True

    def finish_episode(self, proc_idx):
        assert self.current_step[proc_idx] > 0, 'Empty episode detected'
        if self.has_rew_done:
            # Mark the last step as done
            self.history_obs[proc_idx][-1][self.current_episode[proc_idx], self.current_step[proc_idx] - 1, -1] = 1.0
        self.current_episode[proc_idx] += 1
        self.current_step[proc_idx] = 0

        if self.current_episode[proc_idx] >= self.clear_period:
            # Current period is full
            if self.pop_oldest_episode:
                self.pop_episode(proc_idx)
                return False
            else:
                # Start a new one
                self.start_new_period(proc_idx)
                return True

        return False

    def sample_data(self, mini_batch_size, train=True, sample_in_middle=True):
        # Sample a batch of data from the history
        assert sample_in_middle
        if train:
            proc_idx = torch.randint(self.num_opponents, self.num_processes, (mini_batch_size,))
        else:
            proc_idx = torch.randint(0, self.num_opponents, (mini_batch_size,))
        period_idx = [torch.randint(len(self.history_obs[proc_idx[i]]), (1,)).item() for i in range(mini_batch_size)]
        episode_idx = [torch.randint(self.clear_period, (1,)).item()
                       if period_idx[i] != len(self.history_obs[proc_idx[i]]) - 1
                       else torch.randint(self.current_episode[proc_idx[i]] + 1, (1,)).item()
                       for i in range(mini_batch_size)]
        length_idx = [torch.randint(self.history_sizes[proc_idx[i]][period_idx[i]][episode_idx[i]] + 1, (1,)).item()
                      for i in range(mini_batch_size)]
        return (self, (proc_idx, period_idx, episode_idx, length_idx)), proc_idx % self.num_opponents

    def get_episode_mask(self, episode_idx):
        episode_mask = torch.ones((len(episode_idx), self.clear_period), dtype=torch.bool, device=self.device)
        for i, episode in enumerate(episode_idx):
            if self.last_episode_only:
                # Only use the last episode; discard the current one
                if episode > 0:
                    episode_mask[i, episode - 1] = False
            else:
                # Everything until (and including) the current episode
                episode_mask[i, :episode + 1] = False
        return episode_mask

    def get_step_mask(self, proc_idx, period_idx, episode_idx, length_idx):
        step_mask = torch.stack([self.history_step_mask[proc][period] for proc, period in zip(proc_idx, period_idx)])
        step_mask = step_mask.to(self.device)
        for i, (episode, length) in enumerate(zip(episode_idx, length_idx)):
            step_mask[i, episode + 1:] = True
            step_mask[i, episode, length:] = True
        return step_mask

    def get_by_idx(self, proc_idx, period_idx, episode_idx, length_idx):
        obs = torch.stack([self.history_obs[proc][period] for proc, period in zip(proc_idx, period_idx)]).to(self.device)
        act = torch.stack([self.history_act[proc][period] for proc, period in zip(proc_idx, period_idx)]).to(self.device)
        return obs, act, self.get_episode_mask(episode_idx), self.get_step_mask(proc_idx, period_idx, episode_idx, length_idx)

    def get_full_period(self, proc_idx, period_idx):
        obs = torch.stack([self.history_obs[proc][period] for proc, period in zip(proc_idx, period_idx)]).to(self.device)
        act = torch.stack([self.history_act[proc][period] for proc, period in zip(proc_idx, period_idx)]).to(self.device)
        step_mask = torch.stack([self.history_step_mask[proc][period] for proc, period in zip(proc_idx, period_idx)]).to(self.device)
        return obs, act, step_mask

    def get_last_episode(self, proc_idx, period_idx, episode_idx, length_idx):
        obs = torch.stack([self.history_obs[proc][period][episode] for proc, period, episode in zip(proc_idx, period_idx, episode_idx)]).to(self.device)
        act = torch.stack([self.history_act[proc][period][episode] for proc, period, episode in zip(proc_idx, period_idx, episode_idx)]).to(self.device)
        step_mask = torch.zeros((len(episode_idx), self.max_episode_length), dtype=torch.bool, device=self.device)
        for i, length in enumerate(length_idx):
            step_mask[i, length:] = True
        return obs, act, step_mask

    def trim(self):
        # Trim the histories to the storage size
        # This is done by removing the oldest periods
        for proc_idx in range(self.num_processes):
            num_periods = len(self.history_obs[proc_idx])
            num_episodes = (num_periods - 1) * self.clear_period + self.current_episode[proc_idx]
            while num_episodes > self.max_size_per_process:
                assert num_periods > 1, \
                    'Popping the only (incomplete) period in the history; this should not happen'
                self.history_total_size[proc_idx] -= self.history_sizes[proc_idx][0].sum().item()
                self.history_sizes[proc_idx].pop(0)
                self.history_step_mask[proc_idx].pop(0)
                self.history_obs[proc_idx].pop(0)
                self.history_act[proc_idx].pop(0)
                num_periods -= 1
                num_episodes -= self.clear_period

    def get_all_current(self):
        # Get the latest period of every history
        # Additionally return the period index and length index of the retrieved period
        indices = self.get_all_current_indices()
        return ((torch.stack([history_obs[-1] for history_obs in self.history_obs]).to(self.device),
                 torch.stack([history_act[-1] for history_act in self.history_act]).to(self.device),
                 self.get_episode_mask(indices[1]),
                 torch.stack([history_step_mask[-1] for history_step_mask in self.history_step_mask]).to(self.device)),
                indices)

    def get_all_current_indices(self):
        return (torch.tensor([len(history_obs) - 1 for history_obs in self.history_obs]),
                torch.tensor(self.current_episode),
                torch.tensor(self.current_step))

    def clear_to_one_episode(self, proc_idx):
        # Clear the history buffer to a single period, the last non-empty episode, and a new empty episode
        assert self.current_episode[proc_idx] >= 2 and self.current_step[proc_idx] == 0
        # Pop extra periods, this could be unnecessary, but still
        while len(self.history_sizes[proc_idx]) > 1:
            self.history_total_size[proc_idx] -= self.history_sizes[proc_idx][0].sum().item()
            self.history_sizes[proc_idx].pop(0)
            self.history_step_mask[proc_idx].pop(0)
            self.history_obs[proc_idx].pop(0)
            self.history_act[proc_idx].pop(0)

        # Clear the last period
        while self.current_episode[proc_idx] >= 2:
            self.pop_episode(proc_idx)

    def clear(self):
        for i in range(self.num_processes):
            self.history_total_size[i] = 0
            self.history_sizes[i].clear()
            self.history_step_mask[i].clear()
            self.history_obs[i].clear()
            self.history_act[i].clear()
            self.start_new_period(i)


class RolloutStorage(object):
    def __init__(self, num_steps, num_all_agents, obs_shape, action_space,
                 recurrent_hidden_state_size, history_full_size, history_refresh_interval, history_size, sample_size,
                 encoder_max_samples_per_period, self_obs_mode, self_action_mode,
                 step_mode, num_policies, fast_encoder, equal_sampling,
                 joint_training, use_history, leave_on_cpu, has_rew_done, history_use_episodes,
                 use_meta_episode, has_meta_time_step, collect_peer_traj, collect_next_obs,
                 all_has_last_action, include_current_episode, max_episode_length, merge_encoder_computation,
                 use_soft_imitation, last_episode_only, pop_oldest_episode, indices_mapper: AgentIndicesMapper):
        self.obs = torch.zeros(num_steps + 1, num_all_agents, *obs_shape)
        if recurrent_hidden_state_size > 0:
            self.recurrent_hidden_states = torch.zeros(
                num_steps + 1, num_all_agents, recurrent_hidden_state_size)
        else:
            self.recurrent_hidden_states = None
        self.rewards = torch.zeros(num_steps, num_all_agents, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_all_agents, 1)
        self.returns = torch.zeros(num_steps + 1, num_all_agents, 1)
        self.action_log_probs = torch.zeros(num_steps, num_all_agents, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_all_agents, action_shape)
        action_dim = action_shape
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
            action_dim = action_space.n
        self.masks = torch.ones(num_steps + 1, num_all_agents, 1)
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_all_agents, 1)
        if use_soft_imitation:
            self.imp_ratio = torch.ones(num_steps, num_all_agents, 1)
        else:
            self.imp_ratio = None

        self.collect_peer_traj = collect_peer_traj
        if collect_peer_traj:
            num_peers = indices_mapper.args.num_agents - 1
            assert len(obs_shape) == 1
            if all_has_last_action:
                assert obs_shape[0] > action_dim
                self.peer_obs = torch.zeros(num_steps, num_all_agents, num_peers * (obs_shape[0] - action_dim))
            else:
                self.peer_obs = torch.zeros(num_steps, num_all_agents, num_peers * obs_shape[0])
            assert action_shape == 1, f'Unrecognized action shape {action_shape}'
            self.peer_act = torch.zeros(num_steps, num_all_agents, num_peers * action_shape)
            if action_space.__class__.__name__ == 'Discrete':
                self.peer_act = self.peer_act.long()
            self.peer_masks = torch.ones(num_steps, num_all_agents, 1)
        else:
            self.peer_obs = None
            self.peer_act = None
            self.peer_masks = None

        self.collect_next_obs = collect_next_obs
        if collect_next_obs:
            self.next_obs = torch.zeros(num_steps, num_all_agents, *obs_shape)
        else:
            self.next_obs = None

        self.collect_agent_perm = indices_mapper.args.shuffle_agents
        if self.collect_agent_perm:
            # Fill an invalid value here
            self.agent_perm = torch.full((num_steps + 1, num_all_agents, indices_mapper.args.num_agents - 1),
                                         fill_value=indices_mapper.args.num_agents, dtype=torch.long)
        else:
            self.agent_perm = None

        self.indices_mapper = indices_mapper

        self.num_steps = num_steps
        self.step = 0
        self.filled_steps = -1

        self.self_obs_mode = self_obs_mode
        self.self_action_mode = self_action_mode
        self.use_meta_episode = use_meta_episode
        self.clear_period = history_size
        self.episode_counts = torch.zeros(num_all_agents, dtype=torch.long)
        if use_history:
            self.history = PeriodicHistoryStorage(
                num_processes=num_all_agents,
                num_policies=num_policies,
                history_storage_size=history_full_size,
                clear_period=history_size,
                max_samples_per_period=encoder_max_samples_per_period,
                refresh_interval=history_refresh_interval,
                sample_size=sample_size,
                use_episodes=history_use_episodes,
                has_rew_done=has_rew_done,
                has_meta_time_step=has_meta_time_step,
                step_mode=step_mode,
                include_current_episode=include_current_episode,
                obs_shape=obs_shape,
                act_shape=tuple(),
                max_episode_length=max_episode_length,
                merge_encoder_computation=merge_encoder_computation,
                last_episode_only=last_episode_only,
                pop_oldest_episode=pop_oldest_episode,
            )
            self.period_idx = torch.zeros(num_steps, num_all_agents, dtype=torch.long)
            self.episode_idx = torch.zeros(num_steps, num_all_agents, dtype=torch.long)
            self.length_idx = torch.zeros(num_steps, num_all_agents, dtype=torch.long)
        else:
            self.history = self.period_idx = self.episode_idx = self.length_idx = None
        self.num_policies = num_policies
        self.num_all_agents = num_all_agents
        self.fast_encoder = fast_encoder
        self.equal_sampling = equal_sampling
        self.joint_training = joint_training
        self.leave_on_cpu = leave_on_cpu
        self.device = None

    def to(self, device):
        self.device = device
        if self.leave_on_cpu:
            device = 'cpu'
        self.obs = self.obs.to(device)
        if self.recurrent_hidden_states is not None:
            self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        if self.imp_ratio is not None:
            self.imp_ratio = self.imp_ratio.to(device)
        if self.collect_peer_traj:
            self.peer_obs = self.peer_obs.to(device)
            self.peer_act = self.peer_act.to(device)
            self.peer_masks = self.peer_masks.to(device)
        if self.history is not None:
            self.history.to(self.device)

    def current_obs(self):
        return self.obs[self.step].to(self.device)

    def current_rnn_states(self):
        return self.recurrent_hidden_states[self.step].to(self.device) \
            if self.recurrent_hidden_states is not None else None

    def current_masks(self):
        return self.masks[self.step].to(self.device)

    def insert(self, obs, next_obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, infos, period_idx, episode_idx, length_idx,
               imp_ratio, agent_perms):

        # Put stuff in history. This may modify the reward, so do it before reward is put in storage
        if self.history is not None:
            for i, info in enumerate(infos):
                # Add a step
                if self.self_obs_mode:
                    # NOTE: this uses the last obs self.obs[self.step], instead of the added obs,
                    # which is self.obs[self.step + 1]
                    # If this obs is not accompanied by an action from the opponent, then fill a dummy action
                    if self.self_action_mode:
                        self.history.add(i, self.obs[self.step, i], actions[i],
                                         rewards[i][0] if self.history.has_rew_done else None)
                    else:
                        self.history.add(i, self.obs[self.step, i],
                                         info['opponent_act'] if 'opponent_act' in info else None,
                                         rewards[i][0] if self.history.has_rew_done else None)
                else:
                    if 'opponent_obs' in info or 'opponent_act' in info:
                        # assert len(info['opponent_obs']) == len(info['opponent_act'])
                        # add self reward instead of opponent reward
                        self.history.add(i, info['opponent_obs'], info['opponent_act'],
                                         rewards[i][0] if self.history.has_rew_done else None)

                # Optionally finish the episode
                if not masks[i]:
                    if self.self_obs_mode:
                        # In self observation mode, terminal observation might be important (e.g. showdown in KuhnPoker)
                        # Add a raw terminal observation here without opponent action
                        self.history.add(i, torch.from_numpy(info['terminal_observation']).float(), None,
                                         0.0 if self.history.has_rew_done else None)
                    end_period = self.history.finish_episode(i)
                    if self.use_meta_episode:
                        # No truncation whatsoever
                        bad_masks[i] = 1.0
                        # Only done when the meta episode is done
                        masks[i] = 0.0 if end_period else 1.0
        elif self.use_meta_episode:
            for i, info in enumerate(infos):
                if not masks[i]:
                    bad_masks[i] = 1.0
                    self.episode_counts[i] = (self.episode_counts[i] + 1) % self.clear_period
                    masks[i] = 0.0 if self.episode_counts[i] == 0 else 1.0

        self.obs[self.step + 1].copy_(obs)
        if next_obs is not None:
            self.next_obs[self.step].copy_(next_obs)
        else:
            assert self.next_obs is None
        if recurrent_hidden_states is not None:
            self.recurrent_hidden_states[self.step +
                                         1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        if action_log_probs is not None:
            self.action_log_probs[self.step].copy_(action_log_probs)
        if value_preds is not None:
            self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        if imp_ratio is not None:
            self.imp_ratio[self.step].copy_(imp_ratio)
        else:
            assert self.imp_ratio is None
        if period_idx is not None:
            self.period_idx[self.step].copy_(period_idx)
            self.episode_idx[self.step].copy_(episode_idx)
            self.length_idx[self.step].copy_(length_idx)
        else:
            assert self.period_idx is None and self.episode_idx is None and self.length_idx is None

        if self.collect_peer_traj:
            for i, info in enumerate(infos):
                if 'opponent_obs' in info:
                    self.peer_obs[self.step, i].copy_(info['opponent_obs'])
                    self.peer_act[self.step, i].copy_(info['opponent_act'])
                    self.peer_masks[self.step, i].fill_(1.0)
                else:
                    self.peer_masks[self.step, i].fill_(0.0)

        if self.collect_agent_perm:
            self.agent_perm[self.step + 1] = agent_perms
        else:
            assert agent_perms is None

        if self.step > self.filled_steps:
            self.filled_steps = self.step

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        if self.step == 0:
            self.obs[0].copy_(self.obs[-1])
            if self.recurrent_hidden_states is not None:
                self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
            self.masks[0].copy_(self.masks[-1])
            self.bad_masks[0].copy_(self.bad_masks[-1])
            if self.agent_perm is not None:
                self.agent_perm[0].copy_(self.agent_perm[-1])
        if self.history is not None:
            self.history.trim()

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):

        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def sample(self, mini_batch_size, get_history):
        assert self.recurrent_hidden_states is None, "Recurrent policies are not supported by this sampler"
        time_idx = torch.randint(self.filled_steps + 1, size=(mini_batch_size,))
        if self.equal_sampling:
            agent_idx = torch.arange(mini_batch_size) % self.num_all_agents
        else:
            agent_idx = torch.randint(self.num_all_agents, size=(mini_batch_size,))
        obs_batch = self.obs[time_idx, agent_idx].to(self.device)
        action_batch = self.actions[time_idx, agent_idx].to(self.device)
        reward_batch = self.rewards[time_idx, agent_idx].to(self.device)
        nxt_obs_batch = self.obs[time_idx + 1, agent_idx].to(self.device)
        masks_batch = self.masks[time_idx + 1, agent_idx].to(self.device)
        bad_masks_batch = self.bad_masks[time_idx + 1, agent_idx].to(self.device)
        history_batch = self.history.get(agent_idx) if get_history else None
        return obs_batch, action_batch, reward_batch, nxt_obs_batch, \
            masks_batch, bad_masks_batch, agent_idx, history_batch

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_all_agents = self.rewards.size()[0:2]
        batch_size = num_all_agents * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_all_agents, num_steps, num_all_agents * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            if self.equal_sampling:
                # Modify sampled indices to guarantee that all policies receive the same number of samples
                indices = [idx - (idx % self.num_policies) + (i % self.num_policies) for i, idx in enumerate(indices)]
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices].to(self.device)
            if self.next_obs is not None:
                next_obs_batch = self.next_obs.view(-1, *self.next_obs.size()[2:])[indices].to(self.device)
            else:
                next_obs_batch = None
            reward_batch = self.rewards.view(-1, 1)[indices].to(self.device)
            if self.recurrent_hidden_states is not None:
                recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                    -1, self.recurrent_hidden_states.size(-1))[indices].to(self.device)
            else:
                recurrent_hidden_states_batch = None
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices].to(self.device)
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices].to(self.device)
            return_batch = self.returns[:-1].view(-1, 1)[indices].to(self.device)
            masks_batch = self.masks[:-1].view(-1, 1)[indices].to(self.device)
            if self.imp_ratio is not None:
                imp_ratio_batch = self.imp_ratio.view(-1, 1)[indices].to(self.device)
            else:
                imp_ratio_batch = None
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices].to(self.device)
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices].to(self.device)
            agent_indices = torch.tensor(indices) % num_all_agents
            period_batch = self.period_idx.view(-1)[indices] if self.period_idx is not None else None
            episode_batch = self.episode_idx.view(-1)[indices] if self.episode_idx is not None else None
            length_batch = self.length_idx.view(-1)[indices] if self.length_idx is not None else None
            if self.collect_peer_traj:
                peer_obs_batch = self.peer_obs.view(-1, *self.peer_obs.size()[2:])[indices].to(self.device)
                peer_act_batch = self.peer_act.view(-1, self.peer_act.size(-1))[indices].to(self.device)
                peer_masks_batch = self.peer_masks.view(-1, 1)[indices].to(self.device)
            else:
                peer_obs_batch = None
                peer_act_batch = None
                peer_masks_batch = None

            if self.collect_agent_perm:
                agent_perm_batch = self.agent_perm[:-1].view(-1, *self.agent_perm.size()[2:])[indices].to(self.device)
            else:
                agent_perm_batch = None

            yield (obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch,
                   masks_batch, imp_ratio_batch, old_action_log_probs_batch, adv_targ,
                   agent_indices, period_batch, episode_batch, length_batch,
                   peer_obs_batch, peer_act_batch, peer_masks_batch,
                   next_obs_batch, reward_batch, agent_perm_batch)

    def recurrent_generator(self, advantages, use_history, rnn_chunk_length, num_mini_batch=None, mini_batch_size=None):
        assert not use_history

        num_steps, num_all_agents = self.rewards.size()[0:2]
        assert num_steps % rnn_chunk_length == 0, "Number of steps in the buffer must be a multiple of rnn_chunk_length"
        chunks_per_agent = num_steps // rnn_chunk_length
        # Number of chunks
        batch_size = chunks_per_agent * num_all_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) / rollout chunk length ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_all_agents, num_steps, rnn_chunk_length, num_all_agents * num_steps // rnn_chunk_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        else:
            assert mini_batch_size >= rnn_chunk_length, 'There must be at least 1 rollout segment in the mini batch'
            assert mini_batch_size <= batch_size, 'The mini batch size must not be larger than the batch size'
            mini_batch_size = mini_batch_size // rnn_chunk_length

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        obs = _to_chunk_batch(self.obs[:-1], batch_size, rnn_chunk_length)
        rnn_states = _to_chunk_batch(self.recurrent_hidden_states[:-1], batch_size, rnn_chunk_length)
        actions = _to_chunk_batch(self.actions, batch_size, rnn_chunk_length)
        value_preds = _to_chunk_batch(self.value_preds[:-1], batch_size, rnn_chunk_length)
        returns = _to_chunk_batch(self.returns[:-1], batch_size, rnn_chunk_length)
        masks = _to_chunk_batch(self.masks[:-1], batch_size, rnn_chunk_length)
        old_action_log_probs = _to_chunk_batch(self.action_log_probs, batch_size, rnn_chunk_length)
        if advantages is not None:
            advantages = _to_chunk_batch(advantages, batch_size, rnn_chunk_length)
        if self.collect_peer_traj:
            peer_obs = _to_chunk_batch(self.peer_obs, batch_size, rnn_chunk_length)
            peer_act = _to_chunk_batch(self.peer_act, batch_size, rnn_chunk_length)
            peer_masks = _to_chunk_batch(self.peer_masks, batch_size, rnn_chunk_length)
        else:
            peer_obs = None
            peer_act = None
            peer_masks = None

        for indices in sampler:
            obs_batch = _to_rnn_input(obs[indices], self.device)
            recurrent_hidden_states_batch = rnn_states[indices, 0].to(self.device)
            actions_batch = _to_rnn_input(actions[indices], self.device)
            value_preds_batch = _to_rnn_input(value_preds[indices], self.device)
            return_batch = _to_rnn_input(returns[indices], self.device)
            masks_batch = _to_rnn_input(masks[indices], self.device)
            old_action_log_probs_batch = _to_rnn_input(old_action_log_probs[indices], self.device)
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = _to_rnn_input(advantages[indices], self.device)
            agent_indices = (torch.tensor(indices) // chunks_per_agent).repeat(rnn_chunk_length)
            if self.collect_peer_traj:
                peer_obs_batch = _to_rnn_input(peer_obs[indices], self.device)
                peer_act_batch = _to_rnn_input(peer_act[indices], self.device)
                peer_masks_batch = _to_rnn_input(peer_masks[indices], self.device)
            else:
                peer_obs_batch = None
                peer_act_batch = None
                peer_masks_batch = None
            if self.collect_agent_perm:
                agent_perm_batch = self.agent_perm[:-1].view(-1, *self.agent_perm.size()[2:])[indices].to(self.device)
            else:
                agent_perm_batch = None
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, None, old_action_log_probs_batch, adv_targ, agent_indices, None, None, None, \
                peer_obs_batch, peer_act_batch, peer_masks_batch, None, None, agent_perm_batch
