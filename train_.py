import shutil
import pickle
import torch
import os
import time
import wandb
import random
from tqdm import trange
import numpy as np
from learning import utils, algo
from learning.utils import _to_actor_critic_state, PolicyClassificationRewardTracker
from learning.arguments import get_args
from learning.envs import make_vec_envs
from learning.storage_ import RolloutStorage, PeriodicHistoryStorage
from learning.model import LatentPolicy
from evaluation_ import evaluate
from copy import deepcopy


def interpolate_prob(dist, mean_prob, tot_phase, cur_phase):
    dt = 0.0 if dist == 'const' else min(mean_prob, 1.0 - mean_prob)
    ratio = cur_phase / (tot_phase - 1)
    return mean_prob + 2.0 * dt * ratio - dt


def train_embedding(args, train_pool, eval_pool):
    # Supports multiple RL training instances in parallel, potentially with multiple agents in every instance
    # There are num_train_opponents instances, with args.multi_agent agents and num_procs_per_opponent processes
    # in each instance
    # In total, there are args.num_processes * args.multi_agent agents acting in the environments, each with its own
    # rollout buffer; these agents correspond to num_trained_policies models

    assert len(train_pool) == args.train_pool_size
    num_train_opponents = len(train_pool)
    device = 'cuda' if args.cuda else 'cpu'

    # Prepare environments
    # Make (at least one) training environment for every opponent in the pool
    if num_train_opponents > args.num_processes:
        print(f'# processes {args.num_processes} is smaller than pool size {num_train_opponents}, please add more processes')
        quit()
    # Equal number of environments for every opponent
    if args.num_processes % num_train_opponents != 0:
        print(f'# processes {args.num_processes} is not divisible by pool size {num_train_opponents}, '
              f'adjusted # processes is {(args.num_processes // num_train_opponents + 1) * num_train_opponents}')
        args.num_processes = (args.num_processes // num_train_opponents + 1) * num_train_opponents
    num_procs_per_opponent = args.num_processes // num_train_opponents
    num_all_agents = args.num_processes * args.multi_agent
    num_trained_policies = (1 if args.joint_training else num_train_opponents) \
                           * (args.multi_agent if args.separate_model else 1)
    args.num_trained_policies = num_trained_policies
    indices_mapper = utils.AgentIndicesMapper(args)
    print(f'Total # opponents: {num_train_opponents}, \n'
          f'# processes: {args.num_processes}, \n'
          f'# processes per opponent: {num_procs_per_opponent}, \n'
          f'# agents: {num_all_agents}, \n'
          f'# instantiated individual policies: {num_trained_policies}\n')
    print("start making parallel environments")
    envs = make_vec_envs(args, args.env_name, args.seed, args.num_processes, args.log_dir, device, True,
                         always_use_dummy=args.use_dummy_vec_env)
    max_episode_length = envs.env_method('episode_length', indices=0)[0]
    print("finish making parallel environments")
    if args.multi_agent == 1:
        for i in range(args.num_processes):
            # This performs a deepcopy, so every environment receives an exclusive copy
            # Shouldn't matter for SubprocVecEnv, but just to be sure
            envs.env_method('set_opponent',
                            train_pool[i] if i < num_train_opponents else deepcopy(train_pool[i % num_train_opponents]),
                            indices=i)
            if args.env_name == 'Overcooked' or args.env_name == 'MPE':
                envs.env_method('set_id', args.player_id, indices=i)
            if len(train_pool) == 1 and args.env_name == 'Overcooked' and not args.latent_training:
                envs.env_method('set_desire', train_pool[0].ingredient_support_set_id, indices=i)
                if i == 0:
                    print('Single-opponent individual training, setting desire to', train_pool[0].ingredient_support_set_id)
    else:
        for i in range(args.num_processes):
            envs.env_method('set_desire', train_pool[i % num_train_opponents], indices=i)

    use_history = args.latent_training and not (args.identity_encoder or args.emb_encoder)
    use_policy_cls_reward = args.policy_cls_reward_coef is not None
    if use_policy_cls_reward:
        policy_cls_reward_tracker = PolicyClassificationRewardTracker(args, args.num_processes, num_train_opponents)
    else:
        policy_cls_reward_tracker = None

    # Prepare evaluation environments
    num_eval_opponents = len(eval_pool)
    if args.eval_interval is not None:
        eval_envs = make_vec_envs(args, args.env_name, args.seed, num_eval_opponents, args.log_dir, device,
                                  always_use_dummy=args.use_dummy_vec_env, allow_early_resets=True)
        for i in range(num_eval_opponents):
            eval_envs.env_method('set_opponent', eval_pool[i], indices=i)
            if args.env_name == 'Overcooked' or args.env_name == 'MPE':
                eval_envs.env_method('set_id', args.player_id, indices=i)
        if use_history:
            eval_history = PeriodicHistoryStorage(
                num_processes=num_eval_opponents,
                num_policies=num_eval_opponents,
                history_storage_size=args.history_size,
                clear_period=args.history_size,
                max_samples_per_period=None,  # should not be used for training, so place an invalid value here
                refresh_interval=1,
                sample_size=args.sample_size,
                has_rew_done=args.has_rew_done,
                use_episodes=args.history_use_episodes,
                has_meta_time_step=args.has_meta_time_step,
                step_mode=args.step_mode,
                include_current_episode=args.include_current_episode,
                obs_shape=envs.observation_space.shape,
                act_shape=tuple(),
                max_episode_length=max_episode_length,
                merge_encoder_computation=args.merge_encoder_computation,
                last_episode_only=args.last_episode_only,
                pop_oldest_episode=args.pop_oldest_episode,
            )
            eval_history.to(device)
        else:
            eval_history = None
    else:
        eval_envs = eval_history = None

    # Prepare policy model
    base_kwargs = dict(
        hidden_dims=args.hidden_dims,
        act_func=args.act_func
    )
    encoder_kwargs = dict(
        base=args.encoder_base,
        pre_hidden_dims=args.pre_hidden_dims,
        post_hidden_dims=args.post_hidden_dims,
        act_func=args.act_func,
        agg_func=args.agg_func,
        identity_encoder=args.identity_encoder,
        emb_encoder=args.emb_encoder,
        has_rew_done=args.has_rew_done,
        has_meta_time_step=args.has_meta_time_step,

        # Parameters used in attention aggregation layers, even with MLP encoder
        tf_n_heads=args.tf_n_heads,
        tf_dropout=args.tf_dropout,
        tf_pos_emb=args.tf_pos_emb,
        max_episode_length=max_episode_length + 1,
        max_num_episodes=args.history_size
    )
    if args.encoder_base == 'tf':
        encoder_kwargs.update(
            tf_n_layers=args.tf_n_layers,
            tf_hidden_dim=args.tf_hidden_dim,
            tf_ff_dim=args.tf_ff_dim,
            tf_chunk_length=args.tf_chunk_length,
        )
    else:
        encoder_kwargs.update(act_after_agg=args.act_after_agg)
    if len(envs.observation_space.shape) == 3:
        base_kwargs.update(
            hidden_channels=args.hidden_channels,
            kernel_sizes=args.kernel_sizes,
            strides=args.strides,
            paddings=args.paddings
        )
        encoder_kwargs.update(
            hidden_channels=args.hidden_channels,
            kernel_sizes=args.kernel_sizes,
            strides=args.strides,
            paddings=args.paddings
        )
    policy = LatentPolicy(
        algo=args.algo,
        dueling=args.dueling,
        expl_eps=args.expl_eps,
        num_opponents=num_train_opponents,
        policy_cnt=num_trained_policies,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        latent_dim=args.latent_dim,
        discrete_latent=args.discrete_latent,
        quantize_latent=args.quantize_latent,
        deterministic_latent=args.deterministic_latent,
        value_obj=args.value_obj,
        tabular_actor=args.tabular_actor,
        tabular_critic=args.tabular_critic,
        latent_training=args.latent_training,
        use_latent_critic=args.use_latent_critic,
        joint_training=args.joint_training,
        use_aux_pol_cls=args.auxiliary_policy_cls_coef is not None,
        use_aux_value_pred=args.auxiliary_value_pred_coef is not None,
        use_aux_peer_act_pred=args.auxiliary_peer_act_pred_coef is not None,
        use_aux_peer_obs_pred=args.auxiliary_peer_obs_pred_coef is not None,
        indices_mapper=indices_mapper,
        is_recurrent=args.recurrent_policy,
        rnn_hidden_dim=args.rnn_hidden_dim,
        share_actor_critic=args.share_actor_critic,
        contrastive_n_layers=args.contrastive_n_layers,
        contrastive_tau=args.contrastive_tau,
        use_transition_pred=args.auxiliary_transition_pred_coef is not None,
        base_kwargs=base_kwargs,
        encoder_kwargs=encoder_kwargs
    ).to(device)

    # Load pretrained policy, if needed
    if args.pretrained_policy_dir is not None:
        print('Loading pretrained individual policies from', args.pretrained_policy_dir)
        if '%OPP_ID%' in args.pretrained_policy_dir:
            assert len(policy.critics) == num_train_opponents
            if policy.actors is not None:
                assert len(policy.actors) == num_train_opponents
            for i in range(num_train_opponents):
                policy_path = args.pretrained_policy_dir.replace('%OPP_ID%', str(i))
                pretrained_policy = torch.load(policy_path)

                # Only load what we need.
                if policy.actors is not None:
                    assert pretrained_policy.actors is not None and len(pretrained_policy.actors) == 1
                    policy.actors[i].load_state_dict(pretrained_policy.actors[0].state_dict())
                    print(f'Actor {i} loaded')
                else:
                    assert pretrained_policy.actors is None

                assert len(pretrained_policy.critics) == 1
                mismatch = policy.critics[i].load_state_dict(pretrained_policy.critics[0].state_dict(), strict=False)
                print(f'Critic {i} loaded:', mismatch)
        else:
            pretrained_policy = torch.load(args.pretrained_policy_dir)

            # Only load what we need.
            if policy.actors is not None:
                policy.actors.load_state_dict(pretrained_policy.actors.state_dict())
                print('Actors loaded')
            else:
                assert pretrained_policy.actors is None

            mismatch = policy.critics.load_state_dict(pretrained_policy.critics.state_dict(), strict=False)
            print('Critics loaded:', mismatch)

        if policy.actors is not None:
            for p in policy.actors.parameters():
                p.requires_grad_(False)

        for p in policy.critics.parameters():
            p.requires_grad_(False)
    elif args.latent_training:
        # No need for individual actors and critics. Remove them to avoid accidental use.
        policy.actors = policy.critics = None

    if args.pretrained_encoder_dir is not None:
        print('Loading pretrained encoder from', args.pretrained_encoder_dir)
        pretrained_encoder_dict = torch.load(args.pretrained_encoder_dir, map_location=device)
        policy.encoder.load_state_dict(pretrained_encoder_dict)
        print('Encoder loaded')
        for p in policy.encoder.parameters():
            p.requires_grad_(False)

    print('Policy constructed:', policy)

    # Prepare trainers
    assert args.algo == 'ppo', 'Other algorithms are not compatible with periodic history storage'
    if args.latent_training and (args.recon_obj or args.value_obj):
        # Supervised training
        agent = algo.VBPE(
            actor_critic=policy,
            num_epoch=args.num_epochs,
            mini_batch_size=args.mini_batch_size,
            num_mini_batch=args.num_mini_batch,
            num_updates=args.num_updates,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            vqvae_beta_coef=args.vqvae_beta_coef,
            contrastive_coef=args.contrastive_coef,
            fast_encoder=args.fast_encoder,
            recon_obj=args.recon_obj,
            value_obj=args.value_obj,
            value_norm=args.value_norm,
            pcgrad=args.pcgrad,
            log_inside=args.collect_all,
            args=args,
            train_pool=train_pool,
            train_envs=envs,
            eval_pool=eval_pool,
            eval_envs=eval_envs,
            eval_history=eval_history,
            device=device,
            indices_mapper=indices_mapper,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm
        )
        buffer_size = args.num_steps
    elif args.algo == 'ppo':
        assert (not args.latent_training) or args.e2e_obj or (not (args.recon_obj or args.value_obj))
        agent = algo.PPO_(
            actor_critic=policy,
            clip_param=args.clip_param,
            ppo_epoch=args.num_epochs,
            num_mini_batch=args.num_mini_batch,
            rnn_chunk_length=args.rnn_chunk_length,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            kl_coef=args.kl_coef,
            vqvae_beta_coef=args.vqvae_beta_coef,
            contrastive_coef=args.contrastive_coef,
            aux_pol_cls_coef=args.auxiliary_policy_cls_coef,
            aux_val_pred_coef=args.auxiliary_value_pred_coef,
            aux_peer_obs_pred_coef=args.auxiliary_peer_obs_pred_coef,
            aux_peer_act_pred_coef=args.auxiliary_peer_act_pred_coef,
            aux_transition_pred_coef=args.auxiliary_transition_pred_coef,
            encoder_epochs=args.encoder_epochs,
            encoder_updates=args.encoder_updates,
            encoder_mini_batch_size=args.encoder_mini_batch_size,
            fast_encoder=args.fast_encoder,
            value_obj=args.value_obj,
            latent_training=args.latent_training,
            use_history=use_history,
            history_middle_sampling=args.history_middle_sampling,
            pcgrad=args.pcgrad,
            indices_mapper=indices_mapper,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=args.use_clipped_value_loss,
            use_advantage_norm=args.use_advantage_norm
        )
        buffer_size = args.num_steps
    elif args.algo == 'dqn':
        assert not args.latent_training
        assert args.buffer_size % args.num_steps == 0
        agent = algo.DQN(
            actor_critic=policy,
            num_updates=args.num_updates,
            mini_batch_size=args.mini_batch_size,
            gamma=args.gamma,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm
        )
        buffer_size = args.buffer_size
    else:
        raise NotImplementedError(f'Unrecognized training config for algorithm {args.algo}')

    if args.collect_all:
        args.num_steps = buffer_size = args.num_env_steps // args.num_processes

    use_soft_imitation = args.soft_imitation_init_prob is not None

    # Prepare buffers
    # buffer_id % num_trained_policies = policy_id
    if args.load_data_dir is not None:
        print('Loading rollouts from', args.load_data_dir, '...')
        with open(args.load_data_dir, 'rb') as f:
            rollouts = pickle.load(f)
        print('Rollouts loaded, current step position:', rollouts.step)
    else:
        if args.latent_training:
            assert args.clear_history_on_full and args.separate_history
        rollouts = RolloutStorage(
            num_steps=buffer_size,
            num_all_agents=num_all_agents,
            obs_shape=envs.observation_space.shape,
            action_space=envs.action_space,
            recurrent_hidden_state_size=args.rnn_hidden_dim * (1 if args.share_actor_critic else 2),
            history_full_size=args.history_full_size,
            history_refresh_interval=args.history_refresh_interval,
            encoder_max_samples_per_period=args.encoder_max_samples_per_period,
            history_size=args.history_size,
            sample_size=args.sample_size,
            self_obs_mode=args.self_obs_mode,
            self_action_mode=args.self_action_mode,
            step_mode=args.step_mode,
            num_policies=num_trained_policies,
            fast_encoder=args.fast_encoder,
            equal_sampling=args.equal_sampling,
            joint_training=args.joint_training,
            use_history=use_history,
            leave_on_cpu=args.collect_all,
            has_rew_done=args.has_rew_done,
            history_use_episodes=args.history_use_episodes,
            use_meta_episode=args.use_meta_episode,
            has_meta_time_step=args.has_meta_time_step,
            all_has_last_action=args.all_has_last_action,
            collect_peer_traj=args.collect_peer_traj,
            collect_next_obs=args.collect_next_obs,
            include_current_episode=args.include_current_episode,
            max_episode_length=max_episode_length,
            merge_encoder_computation=args.merge_encoder_computation,
            use_soft_imitation=use_soft_imitation,
            last_episode_only=args.last_episode_only,
            pop_oldest_episode=args.pop_oldest_episode,
            indices_mapper=indices_mapper
        )
    assert isinstance(rollouts, RolloutStorage)

    # wandb logging
    if args.wandb_user_name is not None:
        if args.env_name == 'KuhnPoker':
            args.train_optimal_return = sum(p.get_best_response_return() for p in train_pool) / num_train_opponents
            args.eval_optimal_return = sum(p.get_best_response_return() for p in eval_pool) / num_eval_opponents
        run = wandb.init(
            config=args,
            project=args.wandb_project_name or args.env_name,
            entity=args.wandb_user_name,
            notes=args.wandb_comment,
            name=f'{args.algo}_{args.exp_name}_seed{args.seed}',
            dir=args.log_dir
        )
    else:
        run = None

    # Training initialization
    envs.env_method('full_reset')  # Begin the first interaction. The rest will automatically follow, handled by the wrapper
    obs = envs.reset()
    if args.multi_agent == 1:
        assert obs.shape == (args.num_processes, *envs.observation_space.shape), f'{obs.shape} != {(args.num_processes, *envs.observation_space.shape)}'
    else:
        # There are args.num_processes * args.multi_agent agents acting in parallel
        # Map these agents to their buffer_ids
        assert obs.shape == (args.num_processes, args.multi_agent, *envs.observation_space.shape)
        obs = obs.reshape(num_procs_per_opponent, num_train_opponents, *obs.shape[1:])
        obs = obs.transpose(1, 2).reshape(args.multi_agent * args.num_processes, *envs.observation_space.shape)
    rollouts.obs[0].copy_(obs)
    if args.shuffle_agents:
        agent_perm_all = torch.stack(envs.env_method('get_callback_state', 'agent_shuffler'))
        # print(agent_perm_all[0], agent_perm_all[args.train_pool_size])
        rollouts.agent_perm[0].copy_(agent_perm_all)
    rollouts.to(device)

    from collections import deque
    train_stats_by_opponent = {
        k: tuple(deque() for _ in range(num_train_opponents))
        for k in ['reward', 'success_rate', 'visits_per_interaction',
                  'expl_reward_per_interaction', 'expl_reward_per_episode', 'expl_reward_per_step']
    }

    if args.collect_all:
        # Collect all data at once
        num_updates = 1
    else:
        num_updates = args.num_env_steps // args.num_steps // args.num_processes

    all_agent_indices = torch.arange(num_all_agents)
    # print(f'Policy cnt: {len(policy.actors)}, policy indices: {all_agent_indices}')
    phase = int(args.latent_training)

    last_reported_fps_time = time.time()
    last_reported_fps_steps = 0

    if use_soft_imitation:
        soft_imitation_cur_prob = args.soft_imitation_init_prob
        use_br = torch.tensor([
            np.random.rand() < interpolate_prob(args.soft_imitation_prob_dist, soft_imitation_cur_prob,
                                                args.history_size, 0)
            for _ in range(args.num_processes)
        ])
    else:
        use_br = soft_imitation_cur_prob = None

    for j in range(num_updates):
        start = time.time()
        if use_soft_imitation:
            ratio = min(1.0, j * args.num_steps * args.num_processes / args.soft_imitation_decay_steps)
            soft_imitation_cur_prob = args.soft_imitation_init_prob * (1.0 - ratio)
            if args.soft_imitation_init_ppo_clip is not None:
                agent.clip_param = ratio * args.clip_param + (1.0 - ratio) * args.soft_imitation_init_ppo_clip

        if args.policy_cls_reward_decay_steps is not None:
            ratio = min(1.0, j * args.num_steps * args.num_processes / args.policy_cls_reward_decay_steps)
            policy_cls_reward_tracker.reward_coef = args.policy_cls_reward_coef * (1.0 - ratio)

        # Various kinds of scheduling.
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        if args.ent_coef_decay_steps > 0:
            agent.entropy_coef = args.entropy_coef \
                                 * max(1.0 - j * args.num_steps * args.num_processes / args.ent_coef_decay_steps, 0.0)

        if args.expl_decay_steps is not None:
            policy.expl_eps = args.expl_eps + min(j * args.num_steps * args.num_processes / args.expl_decay_steps, 1.0) * (args.expl_eps_final - args.expl_eps)

        # Perform rollouts
        if args.load_data_dir is None:
            for _ in trange(args.num_steps) if args.collect_all else range(args.num_steps):

                # Sample actions
                with torch.no_grad():
                    if phase == 0:
                        value, action, action_log_prob, rnn_states = policy.act(
                            rollouts.current_obs(), rollouts.current_rnn_states(), rollouts.current_masks(),
                            all_agent_indices, None
                        )
                        all_period_idx = all_episode_idx = all_length_idx = imp_ratio = policy_pred = None
                    else:
                        indices = rollouts.history.get_all_current_indices()
                        all_period_idx, all_episode_idx, all_length_idx = indices
                        history = (rollouts.history, (all_agent_indices,) + indices)
                        value, action, action_log_prob, rnn_states = policy.act(
                            rollouts.current_obs(), rollouts.current_rnn_states(), rollouts.current_masks(),
                            all_agent_indices, history, query_ind=args.collect_all
                        )

                        if use_policy_cls_reward:
                            policy_pred = policy.aux_pol_cls_head(policy.last_latents)
                        else:
                            policy_pred = None

                        if use_soft_imitation:
                            # Act with the best responses and compute importance sampling weights
                            imp_ratio = torch.ones(args.num_processes, 1, device=device)
                            br_proc_ids = use_br.nonzero().squeeze(-1)
                            if len(br_proc_ids) > 0:
                                br_latents = policy.last_latents[br_proc_ids]
                                br_obs = rollouts.current_obs()[br_proc_ids]
                                if args.all_has_all_time_steps:
                                    # Remove time steps when calling best responses
                                    br_obs_ = br_obs[..., :-2]
                                else:
                                    br_obs_ = br_obs
                                br_masks = rollouts.current_masks()[br_proc_ids]
                                # Act with the best responses, get the BR actions and mu
                                _, br_action, br_action_log_prob, _ = policy.act(
                                    br_obs_, None, br_masks, br_proc_ids, None, query_ind=True
                                )
                                # Get the actual pi_old and value preds for these BR actions
                                # Here we use the precomputed latents
                                br_value_preds, br_old_action_log_prob, _, _, _, _ = policy.evaluate_actions(
                                    br_obs, None, br_masks, br_proc_ids, None, br_action,
                                    latents=br_latents
                                )
                                # Replace with actual actions, action_log_prob, and value predictions
                                action[br_proc_ids] = br_action
                                action_log_prob[br_proc_ids] = br_action_log_prob  # Log prob from the rollout policy
                                value[br_proc_ids] = br_value_preds
                                # Compute the importance sampling ratio
                                imp_ratio[br_proc_ids] = torch.exp(br_old_action_log_prob - br_action_log_prob)
                                if args.soft_imitation_ratio_clip is not None:
                                    imp_ratio.clamp_(min=1.0 / args.soft_imitation_ratio_clip,
                                                     max=args.soft_imitation_ratio_clip)
                                # print(imp_ratio.min(), imp_ratio.max())
                        else:
                            imp_ratio = None

                # Obser reward and next obs
                if args.multi_agent > 1:
                    action_env = action.reshape(num_procs_per_opponent, args.multi_agent, num_train_opponents, 1)
                    action_env = action_env.transpose(1, 2).reshape(args.num_processes, args.multi_agent, 1)
                else:
                    action_env = action
                obs, reward, done, infos = envs.step(action_env.squeeze(-1))

                if args.policy_cls_reward_coef is not None:
                    with torch.no_grad():
                        policy_cls_reward_tracker.advance(reward, infos, policy_pred,
                                                          rollouts.agent_perm[rollouts.step].T
                                                          if args.shuffle_agents else None)

                # envs.env_method('render', mode='human', indices=0)
                # input('Continue...')
                if args.multi_agent > 1:
                    # Obs & action are truly multi-agent.
                    # Copy reward to all agents. This only works for shared-reward
                    # Done and info (bad_masks) will be handled later
                    obs = obs.reshape(num_procs_per_opponent, num_train_opponents, *obs.shape[1:])
                    obs = obs.transpose(1, 2).reshape(args.multi_agent * args.num_processes, *envs.observation_space.shape)
                    reward = reward.reshape(num_procs_per_opponent, num_train_opponents, 1)
                    reward = reward.repeat(1, args.multi_agent, 1).reshape(args.multi_agent * args.num_processes, 1)

                if args.collect_next_obs:
                    next_obs = obs.clone()
                    for i, info in enumerate(infos):
                        # For finished episodes, the next observation should be the terminal observation
                        if 'episode' in info.keys():
                            next_obs[i].copy_(torch.from_numpy(info['terminal_observation']).float())
                else:
                    next_obs = None

                for i, info in enumerate(infos):

                    if args.visit_reward_coef is not None:
                        # Wrapper sanity check
                        assert ('episode' in info.keys()) == ('episode_stats' in info.keys())
                        assert ('interaction_stats' not in info.keys()) or ('episode_stats' in info.keys())

                    if 'episode' in info.keys():
                        # Episode terminating

                        # Record episode results
                        if not (use_soft_imitation and use_br[i]):
                            train_stats_by_opponent['reward'][i % num_train_opponents].append(info['episode']['r'])
                            if args.env_name == 'Overcooked':
                                train_stats_by_opponent['success_rate'][i % num_train_opponents].append(
                                    info['termination_info'].endswith('completed')
                                )

                        # Record visit stats and exploration rewards
                        if use_history:
                            assert ('interaction_stats' in info) == (rollouts.history.current_episode[i] == args.history_size - 1)
                        if use_policy_cls_reward:
                            train_stats_by_opponent['expl_reward_per_step'][i % num_train_opponents].append(
                                info['expl_reward_per_step']
                            )
                            train_stats_by_opponent['expl_reward_per_episode'][i % num_train_opponents].append(
                                info['expl_reward_per_episode']
                            )
                        if 'interaction_stats' in info:
                            if args.visit_reward_coef is not None:
                                train_stats_by_opponent['visits_per_interaction'][i % num_train_opponents].append(
                                    info['interaction_stats']['visit_cnt']
                                )
                            if use_policy_cls_reward:
                                train_stats_by_opponent['expl_reward_per_interaction'][i % num_train_opponents].append(
                                    info['expl_reward_per_interaction']
                                )

                        if use_soft_imitation:
                            # Decide if the next episode is going to use BR
                            use_br[i] = np.random.rand() < interpolate_prob(
                                args.soft_imitation_prob_dist, soft_imitation_cur_prob,
                                args.history_size, (rollouts.history.current_episode[i] + 1) % args.history_size
                            )

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                if args.multi_agent > 1:
                    masks = masks.reshape(num_procs_per_opponent, num_train_opponents, 1)
                    masks = masks.repeat(1, args.multi_agent, 1).reshape(args.multi_agent * args.num_processes, 1)
                    bad_masks = bad_masks.reshape(num_procs_per_opponent, num_train_opponents, 1)
                    bad_masks = bad_masks.repeat(1, args.multi_agent, 1).reshape(args.multi_agent * args.num_processes, 1)

                if args.shuffle_agents:
                    next_agent_perm_all = torch.stack(envs.env_method('get_callback_state', 'agent_shuffler'))
                else:
                    next_agent_perm_all = None

                rollouts.insert(obs, next_obs, rnn_states, action, action_log_prob, value, reward, masks, bad_masks, infos,
                                all_period_idx, all_episode_idx, all_length_idx, imp_ratio, next_agent_perm_all)

        if args.algo == 'ppo':
            # Get value estimates at the end of the rollout
            with torch.no_grad():
                if args.latent_training:
                    if use_history:
                        indices = rollouts.history.get_all_current_indices()
                        history = (rollouts.history, (all_agent_indices,) + indices)
                    else:
                        history = None
                    latents, _ = policy.encoder.get_latents_and_params(
                        history, all_agent_indices, None, None
                    )
                else:
                    latents = None

                next_value = policy.get_value(rollouts.obs[-1].to(device),
                                              _to_actor_critic_state(
                                                  args.share_actor_critic,
                                                  rollouts.recurrent_hidden_states[-1].to(device)
                                                  if rollouts.recurrent_hidden_states is not None else None
                                              )[1],
                                              rollouts.masks[-1].to(device),
                                              all_agent_indices, latents)[0]

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

        # Also optionally save data
        if args.collect_all:
            raise NotImplementedError('Check the implementation here for periodic history storage.')
            # if args.save_data:
            #     # Write buffers to disk
            #     save_path = os.path.join(args.save_dir, args.algo)
            #     try:
            #         os.makedirs(save_path)
            #     except OSError:
            #         pass
            #     with open(os.path.join(save_path, 'rollouts.pkl'), 'wb') as f:
            #         pickle.dump(rollouts, f)
            #     print("Saved rollouts to {}".format(os.path.join(save_path, 'rollouts.pkl')))

        last_num_steps = j * args.num_processes * args.num_steps
        total_num_steps = last_num_steps + args.num_processes * args.num_steps

        # Training loop
        if args.auxiliary_policy_cls_coef == float('inf') \
                and last_num_steps // args.encoder_update_interval != total_num_steps // args.encoder_update_interval:
            train_info = agent.update_encoder(rollouts)
        else:
            train_info = {}
        warmup_encoder = args.policy_cls_warmup_steps is not None and last_num_steps < args.policy_cls_warmup_steps
        train_info.update(agent.update(rollouts, warmup_polcls=warmup_encoder))

        # Logging, evaluation, saving
        if phase == 0 and args.target_update_period is not None \
                and last_num_steps // args.target_update_period != total_num_steps // args.target_update_period:
            agent.ac_target.load_state_dict(agent.actor_critic.state_dict())

        mean_train_stats_by_opponent = {
            k: None if min(len(train_stats_by_opponent[k][i]) for i in range(num_train_opponents)) == 0
            else tuple(np.mean(train_stats_by_opponent[k][i]) for i in range(num_train_opponents))
            for k in train_stats_by_opponent
        }
        episode_result_ready = mean_train_stats_by_opponent['reward'] is not None

        if args.eval_interval is not None \
                and last_num_steps // args.eval_interval != total_num_steps // args.eval_interval:
            evaluate(args, eval_pool, args.eval_episodes, eval_envs, eval_history, policy,
                     j + phase * num_updates if args.wandb_user_name is not None else None,
                     use_latent=phase, update_history=True)

        end = time.time()
        fps = args.num_steps * args.num_processes / (end - start)
        if episode_result_ready:
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.4f}/{:.4f}, "
                "min/max reward {:.4f}/{:.4f}"
                .format(j, total_num_steps,
                        int(fps),
                        sum(len(train_stats_by_opponent['reward'][i]) for i in range(num_train_opponents)),
                        np.mean(mean_train_stats_by_opponent['reward']), np.median(mean_train_stats_by_opponent['reward']),
                        np.min(mean_train_stats_by_opponent['reward']), np.max(mean_train_stats_by_opponent['reward'])))
            if args.env_name == 'Overcooked':
                print(f'Mean/median success rate: {np.mean(mean_train_stats_by_opponent["success_rate"]):.2f}/{np.median(mean_train_stats_by_opponent["success_rate"]):.2f}, '
                      f'min/max success rate: {np.min(mean_train_stats_by_opponent["success_rate"]):.2f}/{np.max(mean_train_stats_by_opponent["success_rate"]):.2f}')
                if mean_train_stats_by_opponent['visits_per_interaction'] is not None:
                    print('Mean visits per interaction:', np.mean(mean_train_stats_by_opponent['visits_per_interaction']))

        if args.wandb_user_name is not None \
                and last_num_steps // args.log_interval != total_num_steps // args.log_interval \
                and not args.collect_all:
            log_group = 'train' if phase else 'train_ind'
            cur_reported_fps_steps = (j + 1) * args.num_steps * args.num_processes
            cur_reported_fps_time = time.time()
            reported_fps = (cur_reported_fps_steps - last_reported_fps_steps) \
                           / (cur_reported_fps_time - last_reported_fps_time)
            last_reported_fps_time = cur_reported_fps_time
            last_reported_fps_steps = cur_reported_fps_steps
            train_info.update({
                'fps': reported_fps
            })
            if use_history:
                # Compute the average number of episodes in a complete period
                total_episodes_in_history = sum(sum(len(period_sizes) for period_sizes in history_sizes[:-1])
                                                for history_sizes in rollouts.history.history_sizes)
                total_periods_in_history = sum(len(history_sizes) - 1
                                               for history_sizes in rollouts.history.history_sizes)
                if total_periods_in_history > 0:
                    train_info.update({
                        'period_size_mean': total_episodes_in_history / total_periods_in_history,
                    })

            if episode_result_ready:
                train_info.update(reward=np.mean(mean_train_stats_by_opponent['reward']))
                if use_policy_cls_reward:
                    train_info.update(
                        expl_reward_per_episode=np.mean(mean_train_stats_by_opponent['expl_reward_per_episode'])
                    )
                    train_info.update(
                        expl_reward_per_step=np.mean(mean_train_stats_by_opponent['expl_reward_per_step'])
                    )
                    if mean_train_stats_by_opponent['expl_reward_per_interaction'] is not None:
                        train_info.update(
                            expl_reward_per_interaction=np.mean(
                                mean_train_stats_by_opponent['expl_reward_per_interaction']
                            )
                        )
                if args.env_name == 'Overcooked':
                    success_rate = np.mean(mean_train_stats_by_opponent['success_rate'])
                    train_info.update(success_rate=success_rate)
                    if args.save_partial_ckpt is not None:
                        ckpt_id = int(success_rate * args.save_partial_ckpt)
                        save_path = os.path.join(args.save_dir, args.algo)
                        save_file = os.path.join(save_path, f'fcp_{ckpt_id}_{args.save_partial_ckpt}.pt')
                        if ckpt_id > 0 and not os.path.exists(save_file):
                            os.makedirs(save_path, exist_ok=True)
                            torch.save(policy, save_file)
            if mean_train_stats_by_opponent['visits_per_interaction'] is not None:
                train_info.update(
                    visits_per_interaction=np.mean(mean_train_stats_by_opponent['visits_per_interaction'])
                )
            wandb.log({f'{log_group}/{k}': v for k, v in train_info.items()}, step=j + phase * num_updates)
            wandb.log({'env_steps': total_num_steps}, step=j + phase * num_updates)

        # save for every interval steps or for the last epoch
        if args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            os.makedirs(save_path, exist_ok=True)

            if phase == 0:
                torch.save(policy, os.path.join(save_path, 'individual_latest.pt'))
            else:
                torch.save(policy, os.path.join(save_path, "latest.pt"))

            if last_num_steps // args.save_interval != total_num_steps // args.save_interval or j == num_updates - 1:
                torch.save(policy, os.path.join(save_path, f"{total_num_steps}.pt"))
                print('Model saved.')

        # Wrap up the iteration. Clear statistics and prepare buffers for the next rollout
        for k in train_stats_by_opponent:
            for i in range(num_train_opponents):
                train_stats_by_opponent[k][i].clear()
        rollouts.after_update()

    # Finish training
    if args.wandb_user_name is not None:
        run.finish()
    envs.close()
    if eval_envs is not None:
        eval_envs.close()


if __name__ == '__main__':
    arg = get_args()

    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    np.random.seed(arg.seed)
    random.seed(arg.seed)

    if arg.cuda and torch.cuda.is_available() and arg.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    arg.log_dir = os.path.join('./logs', arg.env_name, f'{arg.algo}_{arg.exp_name}_seed{arg.seed}')
    print('Logging to', arg.log_dir)
    if os.path.exists(arg.log_dir):
        response = input(f'Log dir {arg.log_dir} exists, overwrite? ')
        # response = 'n'
        if response in ['y', 'Y', 'yes', 'Yes', 'YES']:
            shutil.rmtree(arg.log_dir)
            print('Directory cleaned.')
        else:
            print('Terminating.')
            quit()
    os.makedirs(arg.log_dir)
    arg.save_dir = arg.log_dir

    torch.set_num_threads(1)

    if arg.env_name == 'Overcooked':
        from environment.overcooked.policy import get_train_eval_pool
    elif arg.env_name == 'KuhnPoker':
        if arg.allow_all_opponents:
            from environment.kuhn_poker.policy_imperfect import get_train_eval_pool
        else:
            from environment.kuhn_poker.policy_new import get_train_eval_pool
    elif arg.env_name == 'MPE':
        from environment.mpe.policy_both import get_train_eval_pool
    else:
        raise NotImplementedError
    train_policies, eval_policies = get_train_eval_pool(arg)
    # THIS STARTS AFTER CREATING THE PARTNER POOLS - TRAINING + EVAL
    if arg.opponent_id is not None:
        train_policies = [train_policies[arg.opponent_id]]
        print(f'Setting a specific opponent {arg.opponent_id} to train against, adjusting train pool size to 1.')
        arg.train_pool_size = 1

    assert len(train_policies) == arg.train_pool_size and len(eval_policies) == arg.eval_pool_size
    # Enforce unique instances in case of stateful policies
    assert len(set(id(pol) for pol in train_policies + eval_policies)) == arg.train_pool_size + arg.eval_pool_size

    # Prepare and check peer ids
    if arg.env_name != 'MPE':
        assert arg.num_agents is None
        arg.num_agents = 2
        arg.policy_id_max = torch.tensor([arg.train_pool_size], dtype=torch.long)
        arg.policy_id_all = torch.arange(arg.train_pool_size).unsqueeze(0)
    else:
        assert all(pol.max_ids == train_policies[0].max_ids for pol in train_policies)
        assert all(pol.max_ids == train_policies[0].max_ids for pol in eval_policies)
        assert arg.num_agents - arg.num_good_agents > 1, 'There must be at least 1 peer predator present'
        if arg.shuffle_agents:
            # Every agent could be predator or prey. Merge their IDs
            arg.policy_id_max = torch.full((arg.num_agents - 1,),
                                           train_policies[0].max_ids[0] + train_policies[0].max_ids[-1],
                                           dtype=torch.long)
            arg.policy_id_all = torch.tensor([pol.current_ids for pol in train_policies]).T
            # Predator IDs precede prey IDs. Add the offset to prey IDs
            arg.policy_id_all[-arg.num_good_agents:] += train_policies[0].max_ids[0]
        else:
            arg.policy_id_max = torch.tensor(train_policies[0].max_ids)
            arg.policy_id_all = torch.tensor([pol.current_ids for pol in train_policies]).T
    assert arg.policy_id_max.shape == (arg.num_agents - 1,)
    assert arg.policy_id_all.shape == (arg.num_agents - 1, arg.train_pool_size)
    assert (arg.policy_id_all < arg.policy_id_max.unsqueeze(-1)).all()

    # train_policies, eval_policies = eval_policies, train_policies
    # arg.train_pool_size, arg.eval_pool_size = arg.eval_pool_size, arg.train_pool_size

    train_embedding(arg, train_policies, eval_policies)
