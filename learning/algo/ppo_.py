import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from learning.storage_ import RolloutStorage
from learning.model import LatentPolicy
from learning.utils import get_latent_losses, pcgrad_modify_gradient, AgentIndicesMapper, permute_agent_ids


class PPO_:
    def __init__(self,
                 actor_critic: LatentPolicy,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 rnn_chunk_length,
                 value_loss_coef,
                 entropy_coef,
                 kl_coef,
                 vqvae_beta_coef,
                 contrastive_coef,
                 aux_pol_cls_coef,
                 aux_val_pred_coef,
                 aux_peer_act_pred_coef,
                 aux_peer_obs_pred_coef,
                 aux_transition_pred_coef,
                 encoder_epochs,
                 encoder_updates,
                 encoder_mini_batch_size,
                 fast_encoder,
                 value_obj,
                 latent_training,
                 use_history,
                 history_middle_sampling,
                 pcgrad,
                 indices_mapper: AgentIndicesMapper,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_advantage_norm=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.rnn_chunk_length = rnn_chunk_length
        self.fast_encoder = fast_encoder
        self.value_obj = value_obj

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.vqvae_beta_coef = vqvae_beta_coef
        self.contrastive_coef = contrastive_coef
        self.aux_pol_cls_coef = aux_pol_cls_coef
        self.aux_val_pred_coef = aux_val_pred_coef
        self.aux_peer_act_pred_coef = aux_peer_act_pred_coef
        self.aux_peer_obs_pred_coef = aux_peer_obs_pred_coef
        self.aux_transition_pred_coef = aux_transition_pred_coef

        self.encoder_epochs = encoder_epochs
        self.encoder_updates = encoder_updates
        self.encoder_mini_batch_size = encoder_mini_batch_size
        self.history_middle_sampling = history_middle_sampling

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_advantage_norm = use_advantage_norm

        self.pcgrad = pcgrad
        self.indices_mapper = indices_mapper

        self.latent_training_mode = latent_training
        self.use_history = use_history

        # Remove parameters that doesn't require gradient
        self.optimizer = optim.Adam([p for p in actor_critic.parameters() if p.requires_grad], lr=lr, eps=eps)

    def encoder_forward(self, batch_history, batch_opp_idx):
        latents, _ = self.actor_critic.encoder(batch_history, None)
        cls_pred = self.actor_critic.aux_pol_cls_head(latents)
        batch_opp_idx = batch_opp_idx.to(cls_pred.device)
        loss = F.cross_entropy(cls_pred, batch_opp_idx, reduction='mean')
        correct_cnt = (cls_pred.argmax(dim=-1) == batch_opp_idx).sum().item()
        return loss, correct_cnt

    def update_encoder(self, rollouts: RolloutStorage):

        total_loss = 0.0
        total_correct = 0

        if self.encoder_epochs is not None:
            total_update_steps = 0
            batch_size = 0
            for _ in range(self.encoder_epochs):
                data_generator = rollouts.history.data_generator(mini_batch_size=self.encoder_mini_batch_size)
                batch_size = 0
                for batch_history, batch_opp_idx in data_generator:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, cur_correct = self.encoder_forward(batch_history, batch_opp_idx)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    total_correct += cur_correct
                    total_update_steps += 1
                    batch_size += len(batch_opp_idx)
            avg_loss = total_loss / total_update_steps
            accuracy = total_correct / (batch_size * self.encoder_epochs)

            # Final validation run
            data_generator = rollouts.history.data_generator(mini_batch_size=self.encoder_mini_batch_size, train=False)
            val_total_loss = 0.0
            val_update_steps = 0
            val_total_correct = 0
            val_batch_size = 0
            with torch.no_grad():
                for batch_history, batch_opp_idx in data_generator:
                    loss, val_cur_correct = self.encoder_forward(batch_history, batch_opp_idx)
                    val_total_loss += loss.item()
                    val_update_steps += 1
                    val_total_correct += val_cur_correct
                    val_batch_size += len(batch_opp_idx)
            val_avg_loss = val_total_loss / val_update_steps
            val_accuracy = val_total_correct / val_batch_size
            encoder_train_info = {
                'sep_aux_pol_cls_batch_size': batch_size,
                'sep_aux_pol_cls_val_batch_size': val_batch_size
            }
        else:
            for _ in range(self.encoder_updates):
                batch_history, batch_opp_idx = rollouts.history.sample_data(
                    self.encoder_mini_batch_size, sample_in_middle=self.history_middle_sampling
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss, cur_correct = self.encoder_forward(batch_history, batch_opp_idx)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_correct += cur_correct
            avg_loss = total_loss / self.encoder_updates
            accuracy = total_correct / (self.encoder_updates * self.encoder_mini_batch_size)

            batch_history, batch_opp_idx = rollouts.history.sample_data(self.encoder_mini_batch_size, train=False)
            with torch.no_grad():
                loss, val_cur_correct = self.encoder_forward(batch_history, batch_opp_idx)
            val_avg_loss = loss.item()
            val_accuracy = val_cur_correct / self.encoder_mini_batch_size

            encoder_train_info = {}

        encoder_train_info.update({
            'sep_aux_pol_cls_loss': avg_loss,
            'sep_aux_pol_cls_acc': accuracy,

            'sep_aux_pol_cls_val_loss': val_avg_loss,
            'sep_aux_pol_cls_val_acc': val_accuracy,
        })
        return encoder_train_info

    def update(self, rollouts: RolloutStorage, warmup_polcls):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if self.use_advantage_norm:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        infos_epoch = {}
        infos_cnt = {}
        update_step = 0
        if self.encoder_updates is not None:
            update_frequency = self.ppo_epoch * self.num_mini_batch // self.encoder_updates
        else:
            update_frequency = None

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.use_history, self.rnn_chunk_length, num_mini_batch=self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, num_mini_batch=self.num_mini_batch
                )

            for sample in data_generator:
                (obs_batch, rnn_states_batch, actions_batch, value_preds_batch,
                 return_batch, masks_batch, imp_ratio_batch, old_action_log_probs_batch, adv_targ,
                 agent_indices, period_batch, episode_batch, length_batch,
                 peer_obs_batch, peer_act_batch, peer_masks_batch,
                 next_obs_batch, reward_batch, agent_perm_batch) = sample

                if peer_masks_batch is not None:
                    peer_masks_batch = peer_masks_batch.squeeze(-1)

                # history_batch = rollouts.history.get_by_idx(agent_indices, period_batch, episode_batch, length_batch)
                history_batch = (rollouts.history, (agent_indices, period_batch, episode_batch, length_batch))

                batch_size = len(obs_batch)

                # Regular PPO, maybe with additional Q-value learning
                values, action_log_probs, dist_entropy, latents, params, features = self.actor_critic.evaluate_actions(
                    obs_batch, rnn_states_batch, masks_batch, agent_indices, history_batch, actions_batch,
                    get_q=self.value_obj
                )

                if self.value_obj:
                    values, qvalue = values
                    qvalue = qvalue[torch.arange(len(qvalue)), actions_batch.squeeze(1)].unsqueeze(1)
                else:
                    qvalue = None

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                if imp_ratio_batch is not None:
                    surr2 = torch.clamp(ratio, (1.0 - self.clip_param) * imp_ratio_batch,
                                        (1.0 + self.clip_param) * imp_ratio_batch) * adv_targ
                else:
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2)

                losses = {}
                infos = {}

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped)
                    if self.value_obj:
                        qvalue_pred_clipped = value_preds_batch + \
                            (qvalue - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        qvalue_losses = (qvalue - return_batch).pow(2)
                        qvalue_losses_clipped = (
                                qvalue_pred_clipped - return_batch).pow(2)
                        losses.update(
                            qvalue_loss=0.5 * torch.max(qvalue_losses, qvalue_losses_clipped)
                        )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2)
                    if self.value_obj:
                        losses.update(
                            qvalue_loss=0.5 * (return_batch - qvalue).pow(2)
                        )

                if self.aux_transition_pred_coef is not None:
                    pred_inputs = torch.cat([
                        self.actor_critic.encoder.convert_inputs(obs_batch, actions_batch.squeeze(-1)),
                        self.actor_critic.last_latents
                    ], dim=-1)
                    pred_features = self.actor_critic.aux_transition_pred_base(pred_inputs)
                    pred_rewards = self.actor_critic.aux_reward_pred_head(pred_features)
                    pred_next_obs = self.actor_critic.aux_next_state_pred_head(pred_features)
                    # assert pred_rewards.shape == reward_batch.shape
                    aux_transition_pred_loss = F.mse_loss(pred_rewards, reward_batch, reduction='none').squeeze(-1) + \
                                                  F.mse_loss(pred_next_obs, next_obs_batch, reduction='none').mean(dim=-1)
                    losses.update(
                        aux_transition_pred_loss=aux_transition_pred_loss
                    )

                if self.aux_peer_act_pred_coef is not None:
                    num_peers = peer_act_batch.shape[-1]
                    aux_peer_act_pred_logits = self.actor_critic.aux_peer_act_pred_head(features)
                    assert aux_peer_act_pred_logits.shape[-1] % num_peers == 0
                    aux_peer_act_pred_logits = torch.split(aux_peer_act_pred_logits,
                                                           aux_peer_act_pred_logits.shape[-1] // num_peers,
                                                           dim=-1)
                    aux_peer_act_pred_losses = [F.cross_entropy(
                        aux_peer_act_pred_logits[i], peer_act_batch[..., i], reduction='none'
                    ) for i in range(num_peers)]
                    aux_peer_act_pred_loss = torch.stack(aux_peer_act_pred_losses).mean(dim=0)
                    losses.update(
                        aux_peer_act_pred_loss=aux_peer_act_pred_loss
                    )

                if self.aux_peer_obs_pred_coef is not None:
                    aux_peer_obs_pred = self.actor_critic.aux_peer_obs_pred_head(features)
                    aux_peer_obs_pred_loss = F.mse_loss(aux_peer_obs_pred, peer_obs_batch, reduction='none').mean(dim=-1)
                    losses.update(
                        aux_peer_obs_pred_loss=aux_peer_obs_pred_loss
                    )

                if self.aux_pol_cls_coef is not None and self.aux_pol_cls_coef != float('inf'):
                    if self.encoder_mini_batch_size is None:
                        # Ues RL mini batch
                        # This will cover all data in the history, so no validation here
                        cls_preds = torch.split(self.actor_critic.aux_pol_cls_head(latents),
                                                self.indices_mapper.args.policy_id_max.tolist(),
                                                dim=-1)
                        opp_indices = self.indices_mapper.to_opponent_indices(agent_indices)

                        # cls_target = opp_indices.to(latents.device)

                        if self.indices_mapper.args.shuffle_agents:
                            cls_targets = permute_agent_ids(self.indices_mapper.args.policy_id_all, opp_indices,
                                                            agent_perm_batch.T)
                        else:
                            assert agent_perm_batch is None
                            cls_targets = [policy_ids[opp_indices].to(latents.device)
                                           for policy_ids in self.indices_mapper.args.policy_id_all]

                        aux_pol_cls_loss = torch.stack([F.cross_entropy(cls_pred, cls_target, reduction='mean')
                                                        for cls_pred, cls_target in zip(cls_preds, cls_targets)]).mean()
                        aux_pol_cls_acc = torch.stack([(cls_pred.argmax(dim=1) == cls_target).float().mean()
                                                       for cls_pred, cls_target in zip(cls_preds, cls_targets)]).mean()
                        losses.update(aux_pol_cls_loss=aux_pol_cls_loss)
                        infos.update(aux_pol_cls_acc=aux_pol_cls_acc)
                    elif update_frequency is None or update_step % update_frequency == 0:
                        # Sample a new mini batch
                        aux_history_batch, aux_opp_idx_batch = rollouts.history.sample_data(
                            self.encoder_mini_batch_size, sample_in_middle=self.history_middle_sampling
                        )
                        aux_pol_cls_loss, aux_pol_cls_correct = self.encoder_forward(aux_history_batch, aux_opp_idx_batch)
                        aux_pol_cls_acc = aux_pol_cls_correct / self.encoder_mini_batch_size
                        # Validation by another mini batch
                        val_history_batch, val_opp_idx_batch = rollouts.history.sample_data(self.encoder_mini_batch_size, train=False)
                        with torch.no_grad():
                            val_pol_cls_loss, val_pol_cls_correct = self.encoder_forward(val_history_batch, val_opp_idx_batch)
                            val_pol_cls_acc = val_pol_cls_correct / self.encoder_mini_batch_size
                            infos.update(aux_pol_cls_val_loss=val_pol_cls_loss.item(), aux_pol_cls_val_acc=val_pol_cls_acc)
                        losses.update(aux_pol_cls_loss=aux_pol_cls_loss)
                        infos.update(aux_pol_cls_acc=aux_pol_cls_acc)

                if self.latent_training_mode:
                    losses.update(get_latent_losses(latents, params, agent_indices,
                                                    get_kl=not (self.actor_critic.encoder.identity_encoder
                                                                or self.actor_critic.encoder.deterministic_latent),
                                                    get_contrastive=False))

                    if self.contrastive_coef > 0.0:
                        opp_indices = self.indices_mapper.to_opponent_indices(agent_indices)
                        losses.update(
                            contrastive_loss=self.actor_critic.get_contrastive_loss(latents, opp_indices)
                        )

                    assert self.aux_val_pred_coef is None
                    # if self.aux_val_pred_coef is not None:
                    #     # NOTE: this doesn't support recurrent policies, for now
                    #     with torch.no_grad():
                    #         val_target = self.actor_critic.get_value(
                    #             obs_batch, agent_indices, None, query_ind=True
                    #         )
                    #     aux_val_pred_loss = self.aux_val_pred_coef * F.mse_loss(
                    #         self.actor_critic.aux_val_pred_head(torch.cat([obs_batch, latents], dim=1)), val_target
                    #     )
                    #     losses.update(aux_val_pred_loss=aux_val_pred_loss)

                # Construct full loss
                value_loss = value_loss.squeeze(-1)
                action_loss = action_loss.squeeze(-1)
                assert value_loss.shape == action_loss.shape == dist_entropy.shape and len(value_loss) == batch_size, \
                    f'{value_loss.shape} {action_loss.shape} {dist_entropy.shape}'
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                if warmup_polcls:
                    # Warmup encoder using policy classification loss only. Detach everything else
                    loss = loss.detach()
                    for ln in losses:
                        if ln != 'aux_pol_cls_loss':
                            losses[ln] = losses[ln].detach()

                if 'qvalue_loss' in losses:
                    assert losses['qvalue_loss'].shape == loss.shape
                    loss += losses['qvalue_loss'] * self.value_loss_coef
                if 'kl_loss' in losses:
                    assert losses['kl_loss'].shape == loss.shape, f'{losses["kl_loss"].shape}'
                    loss += losses['kl_loss'] * self.kl_coef
                if 'vq_loss' in losses:
                    assert losses['vq_loss'].shape == losses['commit_loss'].shape == loss.shape
                    loss += losses['vq_loss'] + losses['commit_loss'] * self.vqvae_beta_coef
                    # The value of commit loss is equal to VQ loss
                    # Delete this so it won't get logged
                    del losses['commit_loss']
                if 'contrastive_loss' in losses:
                    assert losses['contrastive_loss'].shape == loss.shape
                    loss += losses['contrastive_loss'] * self.contrastive_coef
                if 'aux_pol_cls_loss' in losses:
                    assert losses['aux_pol_cls_loss'].numel() == 1
                    loss += self.aux_pol_cls_coef * losses['aux_pol_cls_loss']
                if 'aux_peer_obs_pred_loss' in losses:
                    assert losses['aux_peer_obs_pred_loss'].shape == loss.shape, f'{losses["aux_peer_obs_pred_loss"].shape} {loss.shape}'
                    assert losses['aux_peer_obs_pred_loss'].shape == peer_masks_batch.shape, f'{losses["aux_peer_obs_pred_loss"].shape} {peer_masks_batch.shape}'
                    loss += peer_masks_batch * losses['aux_peer_obs_pred_loss'] * self.aux_peer_obs_pred_coef
                if 'aux_peer_act_pred_loss' in losses:
                    assert losses['aux_peer_act_pred_loss'].shape == loss.shape, f'{losses["aux_peer_act_pred_loss"].shape} {loss.shape}'
                    assert losses['aux_peer_act_pred_loss'].shape == peer_masks_batch.shape, f'{losses["aux_peer_act_pred_loss"].shape} {peer_masks_batch.shape}'
                    loss += peer_masks_batch * losses['aux_peer_act_pred_loss'] * self.aux_peer_act_pred_coef
                if 'aux_transition_pred_loss' in losses:
                    assert losses['aux_transition_pred_loss'].shape == loss.shape, f'{losses["aux_transition_pred_loss"].shape} {loss.shape}'
                    loss += losses['aux_transition_pred_loss'] * self.aux_transition_pred_coef
                processed_loss_list = ['qvalue_loss', 'kl_loss', 'vq_loss', 'contrastive_loss', 'aux_pol_cls_loss',
                                       'aux_peer_obs_pred_loss', 'aux_peer_act_pred_loss', 'aux_transition_pred_loss']
                assert all(k in processed_loss_list for k in losses.keys()), \
                    f'Unprocessed loss found, all loss keys: {losses.keys()}'

                self.optimizer.zero_grad(set_to_none=True)
                if self.pcgrad:
                    loss /= len(loss)
                    policy_indices = self.indices_mapper.to_policy_indices(agent_indices)
                    unique_policies = torch.unique(policy_indices)
                    losses_by_policy = torch.stack([loss[policy_indices == p].sum() for p in unique_policies])
                    pcgrad_modify_gradient(self.actor_critic, losses_by_policy)
                else:
                    loss.mean().backward()

                # from torchviz import make_dot
                # make_dot(loss, params=dict(self.actor_critic.named_parameters())).render('test')
                # print('Dot made')
                # quit()

                infos.update(
                    raw_grad_norm=nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                )
                self.optimizer.step()
                update_step += 1

                infos.update({k: v.mean().item() for k, v in losses.items()})
                infos.update(
                    value_loss=value_loss.mean().item(),
                    action_loss=action_loss.mean().item(),
                    dist_entropy=dist_entropy.mean().item()
                )
                # if self.latent_training_mode:
                #     # _, _, ep_mask, sp_mask = history_batch
                #     ep_mask = rollouts.history.get_episode_mask(episode_batch)
                #     sp_mask = rollouts.history.get_step_mask(agent_indices, period_batch, episode_batch, length_batch)
                #     infos.update(
                #         episode_padding_ratio=(ep_mask.sum() / ep_mask.numel()).item(),
                #         step_padding_ratio=(sp_mask.sum() / sp_mask.numel()).item()
                #     )

                for k in infos:
                    if k not in infos_epoch:
                        infos_epoch[k] = 0.0
                        infos_cnt[k] = 0
                    infos_epoch[k] += infos[k]
                    infos_cnt[k] += 1

        for k in infos_epoch:
            infos_epoch[k] /= infos_cnt[k]

        return infos_epoch
