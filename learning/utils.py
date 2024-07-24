import glob
import os
import torch
import torch.nn as nn
from learning.envs import VecNormalize
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical, Normal
from collections import deque


class AgentIndicesMapper:
    def __init__(self, args):
        self.args = args

    def to_history_indices(self, agent_indices):
        assert self.args.multi_agent == 1, 'WARNING: requesting history indices for self-play training'
        if self.args.separate_history:
            return agent_indices
        return agent_indices % self.args.train_pool_size

    def to_opponent_indices(self, agent_indices):
        return agent_indices % self.args.train_pool_size

    def to_policy_indices(self, agent_indices):
        if self.args.joint_training:
            if self.args.separate_model:
                return (agent_indices // self.args.train_pool_size) % self.args.multi_agent
            return agent_indices % 1
        return agent_indices % self.args.num_trained_policies


def permute_agent_ids(agent_id_per_opponent, opponent_id_per_batch_item, agent_perm_per_batch_item):
    num_other_agents, num_opponents = agent_id_per_opponent.shape
    batch_size, = opponent_id_per_batch_item.shape
    assert agent_perm_per_batch_item.shape == (num_other_agents, batch_size)
    raw_agent_id_per_batch_item = agent_id_per_opponent[:, opponent_id_per_batch_item]  # (num_other_agents, batch_size)
    permuted_agent_id_per_batch_item = torch.gather(raw_agent_id_per_batch_item.to(agent_perm_per_batch_item.device),
                                                    0, agent_perm_per_batch_item)  # same
    return permuted_agent_id_per_batch_item


class PolicyClassificationRewardTracker:
    def __init__(self, args, num_envs, num_opponents):
        self.args = args
        self.num_envs = num_envs
        self.num_opponents = num_opponents
        device = 'cuda' if args.cuda else 'cpu'
        self.reward_coef = args.policy_cls_reward_coef

        self.running_value_stats = torch.zeros(len(args.policy_id_max), num_envs).to(device)
        self.episode_stats = torch.zeros(num_envs).to(device)
        self.interaction_stats = torch.zeros(num_envs).to(device)

    def advance(self, rewards, infos, policy_preds, agent_perms, inspect_idx=None):
        # Compute policy classification reward for each step
        dists = [torch.distributions.Categorical(logits=policy_pred)
                 for policy_pred in torch.split(policy_preds, self.args.policy_id_max.tolist(), dim=-1)]

        if self.args.policy_cls_reward_type == 'accuracy':
            opp_indices = torch.arange(self.num_envs) % self.num_opponents
            if agent_perms is not None:
                permuted_policy_id_all = permute_agent_ids(self.args.policy_id_all, opp_indices, agent_perms)
                # # policy_id_all: (num_agents, num_opponents)
                # # agent_perms: (num_envs, num_agents)
                # # env_policy_id_all: (num_agents, num_envs)
                # env_policy_id_all = self.args.policy_id_all[:, opp_indices]
                # permuted_policy_id_all = torch.gather(env_policy_id_all, 0, agent_perms.T)
                expl_value = torch.stack([dist.probs[torch.arange(self.num_envs), policy_ids]
                                          for dist, policy_ids in zip(dists, permuted_policy_id_all)])
            else:
                expl_value = torch.stack([dist.probs[torch.arange(self.num_envs), policy_ids[opp_indices]]
                                          for dist, policy_ids in zip(dists, self.args.policy_id_all)])
        else:
            expl_value = torch.stack([- dist.entropy() / torch.log(torch.tensor(pid_max, dtype=torch.float32))
                                      for dist, pid_max in zip(dists, self.args.policy_id_max)])

        if inspect_idx is not None:
            print(expl_value[:, inspect_idx])

        if self.args.policy_cls_reward_mode == 'max_full':
            self.running_value_stats = torch.max(self.running_value_stats, expl_value)
            expl_reward = self.running_value_stats
        elif self.args.policy_cls_reward_mode == 'max_diff':
            next_value_stats = torch.max(self.running_value_stats, expl_value)
            expl_reward = next_value_stats - self.running_value_stats
            self.running_value_stats = next_value_stats
        elif self.args.policy_cls_reward_mode == 'diff':
            expl_reward = expl_value - self.running_value_stats
            self.running_value_stats = expl_value
        else:
            expl_reward = expl_value

        expl_reward = expl_reward.mean(dim=0)

        self.episode_stats += expl_reward
        self.interaction_stats += expl_reward

        for i, info in enumerate(infos):
            info['expl_reward'] = expl_reward[i].item()
            if 'episode_stats' in info:
                info['expl_reward_per_episode'] = self.episode_stats[i].item()
                info['expl_reward_per_step'] = self.episode_stats[i].item() / info['episode']['l']
                self.episode_stats[i] = 0.0
                if 'interaction_stats' in info:
                    info['expl_reward_per_interaction'] = self.interaction_stats[i].item()
                    self.interaction_stats[i] = 0.0
                    self.running_value_stats[:, i] = 0.0

        if self.args.policy_cls_reward_coef == float('inf'):
            rewards.copy_(expl_reward.unsqueeze(-1).to(rewards.device))
        else:
            rewards += self.reward_coef * expl_reward.unsqueeze(-1).to(rewards.device)


def pcgrad_modify_gradient(model, losses: torch.Tensor):
    # losses contains loss for every individual policy
    # Modify gradient on model without performing update
    assert model.latent_training_mode
    num_policies = len(losses)

    grads = []
    grads_normed = []
    grads_pc = []
    with_encoder = False if model.encoder is None else None
    for i in range(num_policies):
        # Manually clear gradient instead of calling optim.zero_grad(), since optim.zero_grad()
        # also clears the gradients on individual parts of the model (e.g. critics)
        if model.encoder is not None:
            for p in model.encoder.parameters():
                p.grad = None
        for p in model.actor.parameters():
            p.grad = None

        # Get gradients. Retain graph so the common parts won't be freed
        losses[i].backward(retain_graph=True)

        # Encoder check. In the first training iteration, where the history is empty, encoder may not get gradient
        if with_encoder is None:
            with_encoder = any(p.grad is not None for p in model.encoder.parameters())

        # We only modify the shared-parameter part, ie Encoder & LatentActor
        if with_encoder:
            encoder_grad = [p.grad.flatten() for p in model.encoder.parameters()]
        else:
            encoder_grad = []
            assert model.encoder is None or all(p.grad is None for p in model.encoder.parameters())

        grads.append(torch.cat(encoder_grad + [p.grad.flatten() for p in model.actor.parameters()]))
        # Save a normalized copy to avoid duplicated computation
        grads_normed.append(grads[-1] / torch.dot(grads[-1], grads[-1]))
        grads_pc.append(grads[-1].clone())

        # grads.append(encoder_grad + deque(p.grad.flatten() for p in model.actor.parameters()))
        # grads_normed.append(deque(g / torch.dot(g, g) for g in grads[-1]))
        # grads_pc.append(deque(g.clone() for g in grads[-1]))

    # Compute gradient projections
    def project(idx, gp):
        perm = torch.randperm(num_policies)
        for j in perm:
            if idx == j:
                continue

            dp = torch.dot(gp, grads[j]).item()
            gp -= min(dp, 0.0) * grads_normed[j]

            # for gp, g, gn in zip(grads_pc[i], grads[j], grads_normed[j]):
            #     dp = torch.dot(gp, g)
            #     if dp < 0.0:
            #         gp -= dp * gn
    for i in range(num_policies):
        project(i, grads_pc[i])

    # Assign back modified gradients
    grad_pc = torch.stack(grads_pc, dim=0).sum(dim=0)
    cur_idx = 0
    if with_encoder:
        for p in model.encoder.parameters():
            p.grad.copy_(grad_pc[cur_idx:cur_idx + p.numel()].view_as(p))
            cur_idx += p.numel()
    for p in model.actor.parameters():
        p.grad.copy_(grad_pc[cur_idx:cur_idx + p.numel()].view_as(p))
        cur_idx += p.numel()
    assert grad_pc.numel() == cur_idx, f'Residual gradient elements found: {grad_pc.numel() - cur_idx}'

    # grad_pc = deque(torch.stack(gps, dim=0).sum(dim=0) for gps in zip(*grads_pc))
    # if with_encoder:
    #     for p in model.encoder.parameters():
    #         p.grad.copy_(grad_pc.popleft().view_as(p))
    # for p in model.actor.parameters():
    #     p.grad.copy_(grad_pc.popleft().view_as(p))
    # assert len(grad_pc) == 0


def get_latent_losses(latents, params, policy_indices, get_kl, get_contrastive):
    # Latent-related losses
    # VAE (continuous), VAE (discrete), or VQVAE, probably with contrastive loss
    losses = {}
    if len(params) == 3:
        z_e, z_q, _ = params
        losses.update(
            vq_loss=F.mse_loss(z_e.detach(), z_q, reduction='none'),
            commit_loss=F.mse_loss(z_e, z_q.detach(), reduction='none')
        )
    elif get_kl:
        if len(params) == 1:
            z_dist = Categorical(logits=params[0])
            base_dist = Categorical(logits=torch.zeros_like(params[0]))
        else:
            z_dist = Normal(loc=params[0], scale=(params[1] / 2).exp())
            base_dist = Normal(loc=torch.zeros_like(params[0]), scale=torch.ones_like(params[1]))
        losses.update(
            kl_loss=kl_divergence(z_dist, base_dist).mean(dim=-1)
        )

    if get_contrastive:
        distances = - F.cosine_similarity(latents.unsqueeze(1), latents.unsqueeze(0), dim=-1)
        # distances = ((latents.unsqueeze(1) - latents.unsqueeze(0)) ** 2).mean(dim=2)
        same_pol_mask = policy_indices.unsqueeze(1) == policy_indices.unsqueeze(0)
        losses.update(
            contrastive_loss=(distances[same_pol_mask].sum() - distances[~same_pol_mask].sum()) / len(policy_indices) ** 2
        )
        assert False, 'Contrastive loss doesn\'t support pcgrad, plz check if this is the case; also policy indices passed in may be incorrect now'

    return losses


def remove_diagonal(a):
    # https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
    assert len(a.shape) == 2 and a.shape[0] == a.shape[1]
    n = a.shape[0]
    return a.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def _to_actor_critic_state(share_actor_critic, rnn_state):
    return (rnn_state, rnn_state) if share_actor_critic or rnn_state is None else rnn_state.chunk(2, dim=-1)


def _to_rnn_state(share_actor_critic, actor_state, critic_state):
    return actor_state if share_actor_critic or actor_state is None else torch.cat([actor_state, critic_state], dim=-1)
