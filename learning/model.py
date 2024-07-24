import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from learning.storage_ import PeriodicHistoryStorage
from learning.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical
from learning.utils import init, AgentIndicesMapper, _to_actor_critic_state, _to_rnn_state, remove_diagonal
from vqvae_functions import vq, vq_st


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def aggregate_with_mask(agg_func, x, mask, attn_layer):
    assert len(x.shape) == len(mask.shape) + 1 and x.shape[:-1] == mask.shape
    if agg_func == 'attn':
        return attn_layer(x, mask)
    mask = mask.unsqueeze(-1)
    if agg_func == 'mean':
        return (x * (1.0 - mask.float())).sum(dim=-2) / (1.0 - mask.float()).sum(dim=-2).clamp_(min=1.0)
    if agg_func == 'sum':
        return (x * (1.0 - mask.float())).sum(dim=-2)
    if agg_func == 'max':
        return torch.where(mask, -float('inf'), x).max(dim=-2).values
    raise NotImplementedError


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class CNN(nn.Module):
    cnn_kwargs_list = ['hidden_channels', 'kernel_sizes', 'strides', 'paddings']

    def __init__(self, obs_shape, hidden_channels, kernel_sizes, strides, paddings):
        super().__init__()

        assert len(obs_shape) == 3
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        hidden_channels = [obs_shape[0]] + hidden_channels
        layers = []
        for in_ch, out_ch, k, s, p in zip(hidden_channels[:-1], hidden_channels[1:], kernel_sizes, strides, paddings):
            layers.append(init_(nn.Conv2d(in_ch, out_ch, k, s, p)))
            layers.append(nn.ReLU())
        self.base = nn.Sequential(*layers)
        with torch.no_grad():
            out_shape = self.base(torch.zeros(obs_shape)).shape
            self.out_dim = round(math.prod(out_shape))
            print(f'Conv net built: {obs_shape} -> {out_shape} -> {self.out_dim}')

    def forward(self, x):
        x = self.base(x)
        return x.view(x.shape[:-3] + (self.out_dim,))


class MLP(nn.Module):
    def __init__(self, layer_dims, act_layer_maker, act_at_last):
        super().__init__()

        if len(layer_dims) < 2:
            print(f'Warning: MLP without any linear layer constructed: {layer_dims}')

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            if len(layers) > 0:
                layers.append(act_layer_maker())
            layers.append(init_(nn.Linear(in_dim, out_dim)))
        if act_at_last:
            layers.append(act_layer_maker())
        self.out_dim = layer_dims[-1]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AttentionAggregationLayer(nn.Module):
    def __init__(self, max_seq_len, hidden_dim, n_heads, dropout, pos_emb):
        super().__init__()

        print(f'Building attention aggregation layer with {n_heads} heads, dropout = {dropout}')
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.q = nn.Parameter(torch.randn(1, 1, hidden_dim))
        if pos_emb == 'one_hot':
            self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        else:
            assert pos_emb == 'float' or pos_emb == 'none'
            self.pos_emb = None

    def forward(self, x, mask):
        # x: (*, max_seq_len, hidden_dim)
        # pos_emb: (max_seq_len, hidden_dim)
        # mask: (*, max_seq_len)
        # return: (*, hidden_dim)
        assert x.shape[:-1] == mask.shape, f'{x.shape} != {mask.shape}'
        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.weight.shape, f'{x.shape} != {self.pos_emb.weight.shape}'
            x = x + self.pos_emb.weight
        return self.attn(self.q.expand(len(x), -1, -1), x, x, key_padding_mask=mask, need_weights=False)[0].squeeze(-2)


class AggregatedMLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, pre_hidden_dims, post_hidden_dims, act_func, agg_func, act_after_agg,
                 tf_n_heads, tf_dropout, tf_pos_emb, max_num_episodes, max_episode_length):
        super().__init__()

        if act_func == 'relu':
            act_layer_maker = nn.ReLU
        elif act_func == 'tanh':
            act_layer_maker = nn.Tanh
        else:
            raise NotImplementedError

        pre_hidden_dims = [input_dim] + pre_hidden_dims
        self.act_after_agg = act_after_agg
        self.pre_agg_mlp = MLP(pre_hidden_dims, act_layer_maker, not act_after_agg)

        agg_dim = pre_hidden_dims[-1]
        assert agg_func in ['mean', 'sum', 'max', 'attn']
        self.agg_func = agg_func
        if agg_func == 'attn':
            self.ep_attn = AttentionAggregationLayer(max_episode_length, agg_dim, tf_n_heads, tf_dropout, tf_pos_emb)
            self.final_attn = AttentionAggregationLayer(max_num_episodes, agg_dim, tf_n_heads, tf_dropout, tf_pos_emb)
        else:
            self.ep_attn = self.final_attn = None

        if act_after_agg:
            self.post_agg_act = act_layer_maker()

        self.post_agg_mlp = MLP([agg_dim] + post_hidden_dims + [output_dim], act_layer_maker, False)

    def episode_forward(self, x, sp_mask):
        # Original shape is (*, max_episode_length, .)
        # Output is (*, .), where * stays the same and . may change

        # Mask out empty episodes
        ep_mask = sp_mask.all(dim=-1)
        x_res = aggregate_with_mask(self.agg_func, self.pre_agg_mlp(x[~ep_mask]), sp_mask[~ep_mask], self.ep_attn)
        x = torch.zeros(x.shape[:-2] + x_res.shape[-1:], device=x.device)
        x[~ep_mask] = x_res

        return x

    def period_forward(self, x, ep_mask):
        # Original shape is (*, max_num_episodes, .)
        # Output is (*, .), where * stays the same and . may change
        assert x.shape[:-1] == ep_mask.shape, f'{x.shape} != {ep_mask.shape}'
        x = aggregate_with_mask(self.agg_func, x, ep_mask, self.final_attn)

        if self.act_after_agg:
            x = self.post_agg_act(x)

        return self.post_agg_mlp(x)

    def forward(self, x, ep_mask, sp_mask):
        return self.period_forward(self.episode_forward(x, sp_mask), ep_mask)


class AggregatedAttentionEncoderLayer(nn.Module):
    def __init__(self, max_len, tf_n_layers, tf_n_heads, tf_hidden_dim, tf_ff_dim, tf_dropout, tf_pos_emb, agg_func):
        super().__init__()
        if tf_n_layers > 0:
            print(f'tf_n_layers = {tf_n_layers}, building a transformer')
            self.type = 'tf'
            self.tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(tf_hidden_dim, tf_n_heads, tf_ff_dim, dropout=tf_dropout, batch_first=True),
                tf_n_layers
            )
        else:
            print('tf_n_layers = 0, building a multi-head attention only')
            self.type = 'attn'
            self.attn = nn.MultiheadAttention(tf_hidden_dim, tf_n_heads, dropout=tf_dropout, batch_first=True)
        # One-hot positional encoding
        if tf_pos_emb == 'one_hot':
            self.pos_enc = nn.Embedding(max_len, tf_hidden_dim)
        else:
            assert tf_pos_emb == 'float' or tf_pos_emb == 'none'
            self.pos_enc = None

        self.agg_func = agg_func
        if agg_func == 'attn':
            self.final_attn = AttentionAggregationLayer(max_len, tf_hidden_dim, tf_n_heads, tf_dropout, tf_pos_emb)
        else:
            self.final_attn = None

    def forward(self, x, mask):

        assert x.shape[:-1] == mask.shape, f'{x.shape} != {mask.shape}'
        assert x.shape[-2:] == self.pos_enc.weight.shape, f'{x.shape} != {self.pos_enc.weight.shape}'

        # print('raw:', x.shape, mask.shape)

        # MultiHeadAttention only accepts 3D tensor as input. Flatten all batch dims into a single dim and reshape back
        shape_prefix = x.shape[:-2]
        x = x.view(-1, *x.shape[-2:])
        mask = mask.view(-1, *mask.shape[-1:])

        # print('init reshape:', x.shape, mask.shape)

        # Positional encoding. This should ideally be merged into the attention module
        if self.pos_enc is not None:
            x = x + self.pos_enc.weight

        # if len(shape_prefix) == 1:
        #     print(x, mask)

        if self.type == 'tf':
            x = self.tf(x, src_key_padding_mask=mask)
        else:
            x = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)[0]

        res = aggregate_with_mask(self.agg_func, x, mask, self.final_attn)
        # print('result', res.shape)
        # print('returned', shape_prefix + res.shape[-1:])
        return res.view(*shape_prefix, *res.shape[-1:])


class NestedAttentionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, pre_hidden_dims, post_hidden_dims, act_func, agg_func,
                 tf_n_layers, tf_n_heads, tf_hidden_dim, tf_ff_dim, tf_dropout, tf_chunk_length, tf_pos_emb,
                 max_num_episodes, max_episode_length):
        super().__init__()

        print('NestedAttentionEncoder constructed')

        if act_func == 'relu':
            act_layer_maker = nn.ReLU
        elif act_func == 'tanh':
            act_layer_maker = nn.Tanh
        else:
            raise NotImplementedError

        self.pre_tf_mlp = MLP([input_dim] + pre_hidden_dims + [tf_hidden_dim], act_layer_maker, True)

        self.tf_ep = AggregatedAttentionEncoderLayer(
            max_episode_length, tf_n_layers, tf_n_heads, tf_hidden_dim, tf_ff_dim, tf_dropout, tf_pos_emb, agg_func
        )
        self.tf_h = AggregatedAttentionEncoderLayer(
            max_num_episodes, tf_n_layers, tf_n_heads, tf_hidden_dim, tf_ff_dim, tf_dropout, tf_pos_emb, agg_func
        )
        assert tf_chunk_length is None

        self.post_tf_mlp = MLP([tf_hidden_dim] + post_hidden_dims + [output_dim], act_layer_maker, False)

    def episode_forward(self, x, sp_mask):
        # Filter empty episodes here to prevent NaNs
        ep_mask = sp_mask.all(dim=-1)
        x_res = self.tf_ep(self.pre_tf_mlp(x[~ep_mask]), sp_mask[~ep_mask])
        x = torch.zeros(x.shape[:-2] + x_res.shape[-1:], device=x.device)
        x[~ep_mask] = x_res
        return x

        # return self.tf_ep(self.pre_tf_mlp(x), sp_mask)

    def period_forward(self, x, ep_mask):
        # There can't be any empty periods, so we don't need to filter them
        return self.post_tf_mlp(self.tf_h(x, ep_mask))

    def forward(self, x, ep_mask, sp_mask):
        return self.period_forward(self.episode_forward(x, sp_mask), ep_mask)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, pre_hidden_dims, post_hidden_dims, act_func, agg_func,
                 tf_n_layers, tf_n_heads, tf_hidden_dim, tf_ff_dim, tf_chunk_length):
        super().__init__()

        if act_func == 'relu':
            act_layer_maker = nn.ReLU
        elif act_func == 'tanh':
            act_layer_maker = nn.Tanh
        else:
            raise NotImplementedError

        self.pre_tf_mlp = MLP([input_dim] + pre_hidden_dims + [tf_hidden_dim], act_layer_maker, True)

        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(tf_hidden_dim, tf_n_heads, tf_ff_dim, batch_first=True, dropout=0),
            tf_n_layers
        )
        self.chunk_length = tf_chunk_length

        if agg_func == 'mean':
            self.agg_layer = functools.partial(torch.mean, dim=-2)
        elif agg_func == 'max':
            self.agg_layer = functools.partial(torch.max, dim=-2)
        else:
            raise NotImplementedError

        self.post_tf_mlp = MLP([tf_hidden_dim] + post_hidden_dims + [output_dim], act_layer_maker, False)

    def forward(self, x):
        x = self.pre_tf_mlp(x)
        B, H, D = x.shape
        if self.chunk_length is not None and H > self.chunk_length:
            C = H // self.chunk_length
            x1 = x[:, :C * self.chunk_length, :].view(B * C, self.chunk_length, D)
            x1 = self.tf(x1).view(B, C * self.chunk_length, D)
            if H % self.chunk_length > 0:
                x = torch.cat([x1, self.tf(x[:, C * self.chunk_length:, :])], dim=1)
            else:
                x = x1
        else:
            x = self.tf(x)
        return self.post_tf_mlp(self.agg_layer(x))


class VQEmbedding(nn.Module):
    def __init__(self, quantize_latent, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(quantize_latent, latent_dim)
        self.embedding.weight.data.uniform_(-1./quantize_latent, 1./quantize_latent)

    def forward(self, z_e):
        return vq(z_e, self.embedding.weight)

    def straight_through(self, z_e):
        z_q_st, indices = vq_st(z_e, self.embedding.weight.detach())
        z_q = torch.index_select(self.embedding.weight, dim=0, index=indices)
        return z_q_st, z_q


class Encoder(nn.Module):
    def __init__(self, num_policies, obs_shape, action_space, latent_dim, discrete_latent, quantize_latent,
                 deterministic_latent, indices_mapper: AgentIndicesMapper, has_rew_done, has_meta_time_step,
                 **base_kwargs):
        # Encoder E
        super().__init__()
        if has_rew_done:
            assert len(obs_shape) == 1
            obs_shape = (obs_shape[0] + 2,)
        if has_meta_time_step:
            assert len(obs_shape) == 1
            obs_shape = (obs_shape[0] + 1,)
        self.identity_encoder = base_kwargs['identity_encoder']
        del base_kwargs['identity_encoder']
        self.emb_encoder = base_kwargs['emb_encoder']
        del base_kwargs['emb_encoder']
        self.num_policies = num_policies
        self.latent_dim = latent_dim
        if action_space.__class__.__name__ == "Discrete":
            self.action_dim = action_space.n
            self.discrete_action = True
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            raise NotImplementedError
        self.deterministic_latent = deterministic_latent
        self.indices_mapper = indices_mapper

        if not self.identity_encoder:
            if discrete_latent or quantize_latent > 0 or deterministic_latent:
                latent_output = latent_dim
            else:
                latent_output = latent_dim * 2
            self.output_dim = latent_output
            if not self.emb_encoder:
                assert (len(obs_shape) == 3) == ('hidden_channels' in base_kwargs)
                if len(obs_shape) == 3:
                    cnn_kwargs = {k: base_kwargs[k] for k in CNN.cnn_kwargs_list}
                    for k in CNN.cnn_kwargs_list:
                        del base_kwargs[k]
                    self.cnn = CNN(obs_shape, **cnn_kwargs)
                    obs_features = self.cnn.out_dim
                else:
                    self.cnn = None
                    obs_features = obs_shape[0]

                if base_kwargs['base'] == 'tf':
                    base = NestedAttentionEncoder
                else:
                    base = AggregatedMLPEncoder
                del base_kwargs['base']

                self.base = base(obs_features + self.action_dim, latent_output, **base_kwargs)
            else:
                self.emb = nn.Embedding(num_policies, latent_output)
                # nn.init.normal_(self.emb.weight, 0, 0.1)
                # nn.init.constant_(self.emb.weight[:, latent_dim:], 0.0)
                # nn.init.orthogonal_(self.emb.weight[:, :latent_dim], nn.init.calculate_gain('relu'))

            self.discrete_latent = discrete_latent
            self.quantize_latent = quantize_latent
            if quantize_latent > 0:
                self.codebook = VQEmbedding(quantize_latent, latent_dim)
            else:
                self.codebook = None

    def get_latents_and_params(self, history, agent_indices, latents, params):
        if latents is None:
            # Compute latents directly from histories
            latents, params = self(history, agent_indices)
        elif params is not None:
            # Resample from params
            indices = self.indices_mapper.to_history_indices(agent_indices).to(latents.device)
            # print(f'Resampling {len(latents)} latents with agent indices {agent_indices}, history indices {indices}')
            # input('Continue...')
            latents = latents[indices]
            params = tuple(p[indices] if i < 2 else None for i, p in enumerate(params))
            latents = self.resample(latents, params)
        return latents, params

    def resample(self, z, params):
        # Resample latents from params, override z if needed
        # VQVAE latents are deterministic. Return the original value
        # Otherwise, reconstruct probability distribution and resample
        if not self.identity_encoder and self.quantize_latent == 0 and not self.deterministic_latent:
            if self.discrete_latent:
                logits = params[0]
                z = F.gumbel_softmax(logits, hard=True)
            else:
                mu, logvar = params
                z = torch.distributions.Normal(loc=mu, scale=(logvar / 2).exp()).rsample()
        return z

    def convert_inputs(self, obs, act):
        if self.discrete_action:
            assert act.dtype == torch.long
            # Convert to one-hot

            # -1 in act means no corresponding opponent action
            dummy_action_mask = act == -1
            # Replace with any value so one_hot will not complain
            act = torch.where(dummy_action_mask, 0, act)
            # act[dummy_action_mask] = 0
            act = F.one_hot(act, num_classes=self.action_dim).float()
            # And fill these with zeros
            act[dummy_action_mask] = 0

        if self.cnn is not None:
            obs = self.cnn(obs)

        return torch.cat([obs, act], dim=-1)

    def forward(self, history, agent_indices):
        # history: (obs, act, ep_mask, sp_mask)
        # Return reparameterized sample and distribution parameter

        if self.identity_encoder:
            # Force encoder to be identity (return policy index directly)
            # NOTE: this only works on CUDA
            indices = self.indices_mapper.to_opponent_indices(agent_indices)
            z = F.one_hot(indices.long(), num_classes=self.num_policies).float().cuda()
            params = (z, torch.zeros_like(z))
            return z, params
        if self.emb_encoder:
            # Force encoder to be an embedding
            indices = self.indices_mapper.to_opponent_indices(agent_indices)
            features = self.emb(indices.to(self.emb.weight.device))
        else:
            storage, (proc_idx, period_idx, episode_idx, length_idx) = history
            storage: PeriodicHistoryStorage
            batch_size = len(proc_idx)
            if storage.merge_encoder_computation:
                # Reuse computation for the same period
                # Gather unique inputs
                storage.initialize_cache()
                proc_idx_ = []
                period_idx_ = []
                for i in range(batch_size):
                    if storage.history_episode_cache[proc_idx[i]][period_idx[i]] is None:
                        storage.history_episode_cache[proc_idx[i]][period_idx[i]] = torch.empty(tuple())
                        proc_idx_.append(proc_idx[i])
                        period_idx_.append(period_idx[i])
                obs, act, sp_mask = storage.get_full_period(proc_idx_, period_idx_)
                ep_mask = None
            else:
                obs, act, ep_mask, sp_mask = storage.get_by_idx(proc_idx, period_idx, episode_idx, length_idx)

            history = self.convert_inputs(obs, act)

            if storage.merge_encoder_computation:
                # Compute the shared parts after unique operation
                period_ep_results = self.base.episode_forward(history, sp_mask)
                # Assign back to cache
                j = 0
                for i in range(batch_size):
                    if storage.history_episode_cache[proc_idx[i]][period_idx[i]].shape == torch.Size([]):
                        storage.history_episode_cache[proc_idx[i]][period_idx[i]] = period_ep_results[j]
                        j += 1
                # Compute the unique part
                obs, act, sp_mask = storage.get_last_episode(proc_idx, period_idx, episode_idx, length_idx)
                history = self.convert_inputs(obs, act)
                last_ep_results = self.base.episode_forward(history, sp_mask)

                # Get final results. This is faster than thousands of torch.cat()
                full_ep_results = torch.stack([
                    storage.history_episode_cache[proc_idx[i]][period_idx[i]]
                    for i in range(batch_size)
                ])
                full_ep_results[torch.arange(batch_size), episode_idx] = last_ep_results
                ep_mask = storage.get_episode_mask(episode_idx)

                features = self.base.period_forward(full_ep_results, ep_mask)
            else:
                features = self.base(history, ep_mask, sp_mask)

        if self.deterministic_latent:
            z = F.relu(features)
            params = tuple()
        elif self.quantize_latent > 0:
            z_e = features
            z_q_st, z_q = self.codebook.straight_through(z_e)
            z = z_q_st
            params = (z_e, z_q, None)  # Place an empty value here so the length of params would differ
        elif self.discrete_latent:
            logits = features
            params = (logits, )
            z = F.gumbel_softmax(logits, hard=True)
        else:
            params = features.chunk(2, dim=-1)
            mu, logvar = params
            z = torch.distributions.Normal(loc=mu, scale=(logvar / 2).exp()).rsample()

        return z, params


class Critic(nn.Module):
    def __init__(self, algo, value_obj, dueling, obs_shape, latent_dim, action_dim, cnn_kwargs, hidden_dims,
                 act_layer_maker, tabular_critic, use_rnn, rnn_hidden_dim, rnn_override, cnn_override, base_override):
        super().__init__()

        self.algo = algo
        self.dueling = dueling
        self.tabular_critic = tabular_critic

        if tabular_critic:
            assert not use_rnn, 'RNN not supported for tabular critic'
            if algo == 'ppo' or dueling:
                self.V_table = nn.Embedding(obs_shape[0], 1)
            else:
                self.V_table = None
            if algo == 'dqn' or value_obj:
                self.Q_table = nn.Embedding(obs_shape[0], action_dim)
            else:
                self.Q_table = None
        else:
            last_out_dim = obs_shape[0]

            if cnn_override is not None:
                print('Using overridden CNN in critic')
                self.cnn = cnn_override
            elif cnn_kwargs is not None:
                self.cnn = CNN(obs_shape, **cnn_kwargs)
            else:
                assert len(obs_shape) == 1
                self.cnn = None
            if self.cnn is not None:
                last_out_dim = self.cnn.out_dim

            last_out_dim += latent_dim

            if base_override is not None:
                self.base = base_override
            else:
                self.base = MLP([last_out_dim] + hidden_dims, act_layer_maker, True)
            last_out_dim = self.base.out_dim

            if rnn_override is not None:
                self.rnn = rnn_override
            elif use_rnn:
                self.rnn = RNN(last_out_dim, rnn_hidden_dim)
            else:
                self.rnn = None
            if self.rnn is not None:
                last_out_dim = self.rnn.out_dim

            if algo == 'ppo' or dueling:
                self.V = nn.Linear(last_out_dim, 1)
            else:
                self.V = None
            if algo == 'dqn' or value_obj:
                self.Q = nn.Linear(last_out_dim, action_dim)
            else:
                self.Q = None

    def forward(self, x, rnn_hxs, masks, get_q=False):
        if self.tabular_critic:
            x = x.nonzero()[:, 1]
            if self.algo == 'ppo':
                return (self.V_table(x), self.Q_table(x)) if get_q else self.V_table(x)
            if self.dueling:
                adv = self.Q_table(x)
                return self.V_table(x) + adv - adv.mean(dim=-1, keepdim=True)
            return self.Q_table(x)

        if self.cnn is not None:
            x = self.cnn(x)
        x = self.base(x)
        if self.rnn is not None:
            x, rnn_hxs = self.rnn(x, rnn_hxs, masks)

        return self.get_value_from_features(x, get_q), rnn_hxs

    def get_value_from_features(self, x, get_q=False):
        assert not self.tabular_critic, 'Tabular critic cannot get value from features'
        if self.algo == 'ppo':
            return (self.V(x), self.Q(x)) if get_q else self.V(x)
        if self.dueling:
            adv = self.Q(x)
            return self.V(x) + adv - adv.mean(dim=-1, keepdim=True)
        return self.Q(x)


class LatentActor(nn.Module):
    def __init__(self, obs_shape, action_space, latent_dim, tabular_actor, use_rnn, rnn_hidden_dim, rnn_override,
                 cnn_override, base_override, cnn_kwargs, hidden_dims, act_layer_maker):
        # Latent-conditioned actor D_star
        super().__init__()

        if tabular_actor:
            assert len(obs_shape) == 1 and not use_rnn
            self.table = nn.Embedding(obs_shape[0] * latent_dim, action_space.n)
            self.action_dim = action_space.n
        else:
            self.table = None

            last_out_dim = obs_shape[0]

            if cnn_override is not None:
                self.cnn = cnn_override
            elif len(obs_shape) == 3:
                self.cnn = CNN(obs_shape, **cnn_kwargs)
            else:
                assert len(obs_shape) == 1
                self.cnn = None
            if self.cnn is not None:
                last_out_dim = self.cnn.out_dim

            last_out_dim += latent_dim

            if base_override is not None:
                self.base = base_override
            else:
                self.base = MLP([last_out_dim] + hidden_dims, act_layer_maker, True)
            last_out_dim = hidden_dims[-1]

            if rnn_override is not None:
                self.rnn = rnn_override
            elif use_rnn:
                self.rnn = RNN(last_out_dim, rnn_hidden_dim)
            else:
                self.rnn = None
            if self.rnn is not None:
                last_out_dim = self.rnn.out_dim

            if action_space.__class__.__name__ == "Discrete":
                action_dim = action_space.n
                self.dist = Categorical(last_out_dim, action_dim)
            elif action_space.__class__.__name__ == "Box":
                action_dim = action_space.shape[0]
                self.dist = DiagGaussian(last_out_dim, action_dim)
            elif action_space.__class__.__name__ == "MultiBinary":
                action_dim = action_space.shape[0]
                self.dist = Bernoulli(last_out_dim, action_dim)
            else:
                raise NotImplementedError
            self.action_dim = action_dim

    def act(self, inputs, rnn_hxs, masks, latents, deterministic=False):
        dist, rnn_hxs, features = self.get_action_dist(inputs, rnn_hxs, masks, latents)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return action, action_log_probs, rnn_hxs, features

    def get_action_dist(self, inputs, rnn_hxs, masks, latents):
        if self.table is not None:
            idx = inputs.argmax(dim=-1) * latents.shape[-1] + latents.argmax(dim=-1)
            dist = FixedCategorical(logits=self.table(idx))
            rnn_hxs = features = None
        else:
            if self.cnn is not None:
                inputs = self.cnn(inputs)

            if latents is not None:
                inputs = torch.cat([inputs, latents], dim=-1)

            features = self.base(inputs)

            if self.rnn is not None:
                features, rnn_hxs = self.rnn(features, rnn_hxs, masks)

            dist = self.dist(features)

        return dist, rnn_hxs, features

    def evaluate_actions(self, inputs, rnn_hxs, masks, latents, action):
        dist, _, features = self.get_action_dist(inputs, rnn_hxs, masks, latents)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return action_log_probs, dist_entropy, features


class LatentPolicy(nn.Module):
    def __init__(self, algo, dueling, expl_eps, num_opponents, policy_cnt, obs_shape, action_space, latent_dim,
                 discrete_latent, quantize_latent, deterministic_latent, value_obj, tabular_actor, tabular_critic,
                 latent_training, use_latent_critic, joint_training, use_aux_pol_cls, use_aux_value_pred,
                 indices_mapper: AgentIndicesMapper, is_recurrent, rnn_hidden_dim, share_actor_critic,
                 use_aux_peer_act_pred, use_aux_peer_obs_pred, contrastive_n_layers, contrastive_tau,
                 use_transition_pred, base_kwargs, encoder_kwargs):
        super().__init__()

        self.is_recurrent = is_recurrent
        self.rnn_hidden_dim = rnn_hidden_dim
        self.latent_training_mode = latent_training
        self.num_opponents = num_opponents
        self.num_policies = policy_cnt
        self.joint_training = joint_training
        self.share_actor_critic = share_actor_critic
        self.algo = algo
        self.expl_eps = expl_eps
        self.indices_mapper = indices_mapper
        if contrastive_n_layers is None:
            self.contrastive_proj_head = None
        else:
            contrastive_layers = []
            for i in range(contrastive_n_layers):
                contrastive_layers.append(nn.Linear(latent_dim, latent_dim, bias=False))
                contrastive_layers.append(nn.ReLU())
            contrastive_layers.append(nn.Linear(latent_dim, latent_dim, bias=False))
            self.contrastive_proj_head = nn.Sequential(*contrastive_layers)
        self.contrastive_tau = contrastive_tau

        # E
        if latent_training:
            self.encoder = Encoder(policy_cnt, obs_shape, action_space, latent_dim, discrete_latent, quantize_latent,
                                   deterministic_latent, indices_mapper, **encoder_kwargs)
            print('Encoder constructed:')
            print(self.encoder)
        else:
            self.encoder = None
        self.last_latents = None

        # Parse base args
        # Always contains an MLP, optionally contains an CNN up front for image inputs
        assert (len(obs_shape) == 3) == ('hidden_channels' in base_kwargs)
        if len(obs_shape) == 3:
            # Contains CNN
            cnn_kwargs = {k: base_kwargs[k] for k in CNN.cnn_kwargs_list}
            for k in CNN.cnn_kwargs_list:
                del base_kwargs[k]
        else:
            cnn_kwargs = None

        hidden_dims = base_kwargs['hidden_dims']
        del base_kwargs['hidden_dims']

        if base_kwargs['act_func'] == 'relu':
            act_layer_maker = nn.ReLU
        elif base_kwargs['act_func'] == 'tanh':
            act_layer_maker = nn.Tanh
        else:
            raise NotImplementedError
        del base_kwargs['act_func']

        assert len(base_kwargs) == 0, f'The following keys in base config can not be parsed: {base_kwargs}'

        # Latent actor and critic
        if latent_training:
            self.actor = LatentActor(obs_shape, action_space, latent_dim, tabular_actor,
                                     is_recurrent, rnn_hidden_dim, None, None, None,
                                     cnn_kwargs, hidden_dims, act_layer_maker)
        else:
            self.actor = None

        self.use_latent_critic = use_latent_critic
        if use_latent_critic:
            self.critic = Critic(algo, value_obj, dueling, obs_shape, self.encoder.latent_dim, action_space.n,
                                 cnn_kwargs, hidden_dims, act_layer_maker, tabular_critic,
                                 is_recurrent, rnn_hidden_dim,
                                 self.actor.rnn if share_actor_critic else None,
                                 self.actor.cnn if share_actor_critic else None,
                                 self.actor.base if share_actor_critic else None)
        else:
            self.critic = None

        # Separate actors and critics
        if encoder_kwargs['tf_pos_emb'] == 'float':
            # Remove time steps for loading individual actors and critics
            br_obs_shape = (obs_shape[0] - 2,)
        else:
            br_obs_shape = obs_shape

        self.critics = nn.ModuleList([
            Critic(algo, value_obj, dueling, br_obs_shape, 0, action_space.n,
                   cnn_kwargs, hidden_dims, act_layer_maker, tabular_critic,
                   is_recurrent, rnn_hidden_dim, None, None, None)
            for _ in range(policy_cnt)
        ])
        if algo == 'ppo':
            self.actors = nn.ModuleList([
                LatentActor(br_obs_shape, action_space, 0, False, is_recurrent, rnn_hidden_dim,
                            self.critics[i].rnn if share_actor_critic else None,
                            self.critics[i].cnn if share_actor_critic else None,
                            self.critics[i].base if share_actor_critic else None,
                            cnn_kwargs, hidden_dims, act_layer_maker)
                for i in range(policy_cnt)
            ])
        else:
            self.actors = None

        # Auxiliary tasks
        if use_aux_pol_cls:
            self.aux_pol_cls_head = nn.Linear(self.encoder.latent_dim, indices_mapper.args.policy_id_max.sum().item())
        else:
            self.aux_pol_cls_head = None

        if use_aux_value_pred:
            self.aux_val_pred_head = Critic(algo, value_obj, dueling, obs_shape, self.encoder.latent_dim, action_space.n,
                                            cnn_kwargs, hidden_dims, act_layer_maker, tabular_critic,
                                            is_recurrent, rnn_hidden_dim, None, None, None)
        else:
            self.aux_val_pred_head = None

        if use_aux_peer_act_pred:
            feature_dim = rnn_hidden_dim if is_recurrent else hidden_dims[-1]
            assert is_recurrent == use_aux_peer_obs_pred
            num_peers = indices_mapper.args.num_agents - 1
            self.aux_peer_act_pred_head = nn.Linear(feature_dim, num_peers * action_space.n)
        else:
            self.aux_peer_act_pred_head = None

        if use_aux_peer_obs_pred:
            assert is_recurrent, 'RNN should be activated for LIAM'
            feature_dim = rnn_hidden_dim
            # LIAM has past action in the observation space, but not for the opponent
            num_peers = indices_mapper.args.num_agents - 1
            self.aux_peer_obs_pred_head = nn.Linear(feature_dim, num_peers * (obs_shape[0] - action_space.n))
        else:
            self.aux_peer_obs_pred_head = None

        if use_transition_pred:
            in_dim = obs_shape[0] + action_space.n + self.encoder.latent_dim
            self.aux_transition_pred_base = MLP([in_dim] + hidden_dims, act_layer_maker, True)
            self.aux_reward_pred_head = nn.Linear(hidden_dims[-1], 1)
            self.aux_next_state_pred_head = nn.Linear(hidden_dims[-1], obs_shape[0])
        else:
            self.aux_transition_pred_base = None
            self.aux_reward_pred_head = None
            self.aux_next_state_pred_head = None

    def get_contrastive_features(self, latents):
        return self.contrastive_proj_head(latents)

    def get_contrastive_loss(self, latents, indices):
        # NT-Xent loss as described in SimCLR
        # Modified to account for (potentially) multiple positives

        # Get scaled cosine similarities
        assert len(latents.shape) == 2 and latents.shape[:1] == indices.shape
        features = self.get_contrastive_features(latents)
        cosines = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1) / self.contrastive_tau

        # Get same-class mask
        mask = indices.unsqueeze(1) == indices.unsqueeze(0)

        # Remove diagonals
        cosines = remove_diagonal(cosines)
        mask = remove_diagonal(mask).float().to(cosines.device)

        # Compute multi-label NT-Xent loss
        positives = mask.sum(dim=-1).clamp(min=1.0)
        loss = F.cross_entropy(cosines, mask / positives.unsqueeze(-1), reduction='none')

        return loss

    def get_qvalue(self, inputs, rnn_hxs, masks, agent_indices):
        # WARNING: UNTESTED WITH RNN
        qvalue = torch.zeros(len(inputs), self.actor.action_dim, device=inputs.device)
        indices = self.indices_mapper.to_policy_indices(agent_indices)
        assert indices.min().item() >= 0
        assert indices.max().item() < len(self.critics)
        for i, critic in enumerate(self.critics):
            mask = indices == i
            if mask.any():
                rhs = rnn_hxs[mask[:len(rnn_hxs)]] if rnn_hxs is not None else None
                qvalue[mask] = critic(inputs[mask], rhs, masks[mask])
        return qvalue

    def act(self, inputs, rnn_hxs, masks, agent_indices, history, deterministic=False,
            latents=None, params=None, query_ind=False):
        # Takes and returns full RNN states
        if self.algo == 'dqn':
            # WARNING: UNTESTED WITH RNN
            qvalue = self.get_qvalue(inputs, rnn_hxs, masks, agent_indices)
            action = qvalue.argmax(dim=-1)
            if not deterministic:
                mask = torch.rand(len(inputs)) < self.expl_eps
                if mask.any():
                    action[mask] = torch.randint(self.actor.action_dim, size=(mask.sum(),), device=action.device)
            return None, action.unsqueeze(-1), None

        actor_rnn_hxs, critic_rnn_hxs = _to_actor_critic_state(self.share_actor_critic, rnn_hxs)

        if self.latent_training_mode and not query_ind:
            latents, params = self.encoder.get_latents_and_params(history, agent_indices, latents, params)
            self.last_latents = latents
            action, action_log_probs, nxt_actor_rnn_hxs, features = self.actor.act(
                inputs, actor_rnn_hxs, masks, latents, deterministic=deterministic
            )
            value = self.critic.get_value_from_features(features) if self.share_actor_critic else None
        else:
            action = torch.zeros(len(inputs), 1, dtype=torch.long, device=inputs.device)
            action_log_probs = torch.zeros(len(inputs), 1, device=inputs.device)
            nxt_actor_rnn_hxs = torch.zeros_like(actor_rnn_hxs) if self.is_recurrent else None
            value = torch.zeros(len(inputs), 1, device=inputs.device) if self.share_actor_critic else None
            indices = self.indices_mapper.to_policy_indices(agent_indices)
            assert indices.min().item() >= 0
            assert indices.max().item() < len(self.actors)
            for i, actor in enumerate(self.actors):
                mask = indices == i
                if mask.any():
                    rhs = actor_rnn_hxs[mask[:len(actor_rnn_hxs)]] if actor_rnn_hxs is not None else None
                    act, alp, feat, rhs = actor.act(inputs[mask], rhs, masks[mask], None,
                                                    deterministic=deterministic)
                    action[mask] = act
                    action_log_probs[mask] = alp
                    if self.is_recurrent:
                        nxt_actor_rnn_hxs[mask[:len(actor_rnn_hxs)]] = rhs
                    if self.share_actor_critic:
                        value[mask] = self.critics[i].get_value_from_features(feat)

        if self.share_actor_critic:
            nxt_critic_rnn_hxs = nxt_actor_rnn_hxs
        else:
            value, nxt_critic_rnn_hxs = self.get_value(
                inputs, critic_rnn_hxs, masks, agent_indices, latents, query_ind=query_ind
            )

        nxt_rnn_hxs = _to_rnn_state(self.share_actor_critic, nxt_actor_rnn_hxs, nxt_critic_rnn_hxs)

        return value, action, action_log_probs, nxt_rnn_hxs

    def get_action_dist(self, inputs, rnn_hxs, masks, agent_indices, history, latents=None, params=None):
        # Takes full RNN states, doesn't return any
        actor_rnn_hxs = _to_actor_critic_state(self.share_actor_critic, rnn_hxs)[0]

        if self.latent_training_mode:
            latents, params = self.encoder.get_latents_and_params(history, agent_indices, latents, params)
            action_dist, _, _ = self.actor.get_action_dist(inputs, actor_rnn_hxs, masks, latents)
        else:
            if self.algo == 'dqn':
                # WARNING: UNTESTED WITH RNN
                qvalue = self.get_qvalue(inputs, rnn_hxs, masks, agent_indices)
                action_probs = torch.zeros_like(qvalue)
                action_probs[torch.arange(len(action_probs)), qvalue.argmax(dim=-1)] = 1.0
                action_dist = FixedCategorical(probs=action_probs)
            else:
                logits = torch.zeros(len(inputs), self.actors[0].action_dim, device=inputs.device)
                indices = self.indices_mapper.to_policy_indices(agent_indices)
                assert indices.min().item() >= 0
                assert indices.max().item() < len(self.actors)
                for i, actor in enumerate(self.actors):
                    mask = indices == i
                    if mask.any():
                        rhs = actor_rnn_hxs[mask[:len(actor_rnn_hxs)]] if actor_rnn_hxs is not None else None
                        logits[mask] = actor.get_action_dist(inputs[mask], rhs, masks[mask], None)[0].logits
                action_dist = FixedCategorical(logits=logits)
            latents = params = None
        return action_dist, latents, params

    def get_value(self, inputs, critic_rnn_hxs, masks, agent_indices, latents, get_q=False, query_ind=False):
        # Takes and returns critic RNN states
        if self.algo == 'dqn':
            # WARNING: UNTESTED WITH RNN
            return (None, self.get_qvalue(inputs, critic_rnn_hxs, masks, agent_indices)) if get_q else None

        if self.use_latent_critic and not query_ind:
            assert get_q is False, 'Latent critic does not support q-value'
            return self.critic(torch.cat([inputs, latents], dim=-1), critic_rnn_hxs, masks)

        value = torch.zeros(len(inputs), 1, device=inputs.device)
        qvalue = torch.zeros(len(inputs), self.actor.action_dim, device=inputs.device) if get_q else None
        nxt_critic_rnn_hxs = torch.zeros_like(critic_rnn_hxs) if self.is_recurrent else None
        indices = self.indices_mapper.to_policy_indices(agent_indices)
        assert indices.min().item() >= 0
        assert indices.max().item() < len(self.critics)
        for i, critic in enumerate(self.critics):
            mask = indices == i
            if mask.any():
                rhs = critic_rnn_hxs[mask[:len(critic_rnn_hxs)]] if critic_rnn_hxs is not None else None
                if get_q:
                    (v, q), rhs = critic(inputs[mask], rhs, masks[mask], get_q=True)
                    value[mask] = v
                    qvalue[mask] = q
                else:
                    value[mask], rhs = critic(inputs[mask], rhs, masks[mask])
                if self.is_recurrent:
                    nxt_critic_rnn_hxs[mask[:len(nxt_critic_rnn_hxs)]] = rhs
        return ((value, qvalue) if get_q else value), nxt_critic_rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, agent_indices, history, action,
                         latents=None, params=None, get_q=False):
        # Takes full RNN states, doesn't return any
        actor_rnn_hxs, critic_rnn_hxs = _to_actor_critic_state(self.share_actor_critic, rnn_hxs)
        del rnn_hxs

        if self.latent_training_mode:
            latents, params = self.encoder.get_latents_and_params(history, agent_indices, latents, params)
            self.last_latents = latents
            action_log_probs, dist_entropy, features = self.actor.evaluate_actions(
                inputs, actor_rnn_hxs, masks, latents, action
            )
            if self.share_actor_critic:
                value = self.critic.get_value_from_features(features)
            else:
                value = None
        else:
            action_log_probs = torch.zeros(len(inputs), 1, device=inputs.device)
            dist_entropy = torch.zeros(len(inputs), device=inputs.device)
            if self.share_actor_critic:
                value = torch.zeros(len(inputs), 1, device=inputs.device)
            else:
                value = None
            features = None
            indices = self.indices_mapper.to_policy_indices(agent_indices)
            assert indices.min().item() >= 0
            assert indices.max().item() < len(self.actors)
            for i, actor in enumerate(self.actors):
                mask = indices == i
                if mask.any():
                    rhs = actor_rnn_hxs[mask[:len(actor_rnn_hxs)]] if actor_rnn_hxs is not None else None
                    alp, de, feat = actor.evaluate_actions(inputs[mask], rhs, masks[mask], None, action[mask])
                    action_log_probs[mask] = alp
                    dist_entropy[mask] = de
                    if self.share_actor_critic:
                        value[mask] = self.critics[i].get_value_from_features(feat)
                    if features is None:
                        features = torch.zeros(len(inputs), feat.size(-1), device=inputs.device)
                    features[mask] = feat
            latents = params = None

        if not self.share_actor_critic:
            value, _ = self.get_value(inputs, critic_rnn_hxs, masks, agent_indices, latents, get_q=get_q)

        return value, action_log_probs, dist_entropy, latents, params, features


class RNN(nn.Module):
    def __init__(self, recurrent_input_size, hidden_size):
        super(RNN, self).__init__()

        self.out_dim = hidden_size
        self.gru = nn.GRU(recurrent_input_size, hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
