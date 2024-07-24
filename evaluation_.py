import numpy as np
import torch
import wandb
from learning.storage_ import PeriodicHistoryStorage
from learning.model import LatentPolicy
from learning.utils import PolicyClassificationRewardTracker
# from environment.kuhn_poker.utils import get_probs_and_return
from time import sleep


def evaluate(args, eval_policies, eval_episodes, eval_envs, eval_history: PeriodicHistoryStorage, policy: LatentPolicy,
             log_steps, use_latent, update_history, inspect_idx=None):

    dump_latents = hasattr(args, 'dump_latents') and args.dump_latents

    if eval_history is not None:
        assert args.history_use_episodes and (eval_episodes == eval_history.clear_period or args.pop_oldest_episode or args.reward_drop_ratio is not None)
    num_eval_policies = len(eval_policies)
    assert num_eval_policies == eval_envs.num_envs

    eval_envs.env_method('full_reset')
    last_obs = obs = eval_envs.reset()
    update_history = update_history and eval_history is not None
    if update_history:
        eval_history.clear()

    assert args.multi_agent == 1, 'Evaluation only works for single-agent training'
    all_agent_indices = torch.arange(num_eval_policies)
    if policy.is_recurrent:
        rnn_states = torch.zeros(num_eval_policies, policy.rnn_hidden_dim * (1 if policy.share_actor_critic else 2)).to(obs.device)
    else:
        rnn_states = None
    masks = torch.zeros(num_eval_policies, 1).to(obs.device)
    done = np.array([True] * num_eval_policies)

    eval_stats_by_opponent = {
        k: tuple([] for _ in range(num_eval_policies))
        for k in ['reward', 'success_rate', 'visits_per_interaction', 'showdown', 'latents']
    }

    use_policy_cls_reward = args.policy_cls_reward_coef is not None and inspect_idx is not None
    if use_policy_cls_reward:
        assert args.policy_cls_reward_coef == 0.0
    policy_cls_reward_tracker = PolicyClassificationRewardTracker(args, num_eval_policies, num_eval_policies) \
        if use_policy_cls_reward else None
    first_inspect = True

    while min(len(eval_stats_by_opponent['reward'][i]) for i in range(num_eval_policies)) < eval_episodes:
        if inspect_idx is not None:
            render_kwargs = {
                'indices': inspect_idx
            }
            if args.env_name == 'MPE':
                render_kwargs['close'] = False
            first_inspect = False
            if not first_inspect:
                eval_envs.env_method('render', 'human', **render_kwargs)
            else:
                first_inspect = False
            if done[inspect_idx]:
                input('New episode starting, press enter to continue...')
            else:
                sleep(0.2)

        if dump_latents:
            for i in range(num_eval_policies):
                if done[i]:
                    eval_stats_by_opponent['latents'][i].append([])

        # Sample actions
        with torch.no_grad():
            if eval_history is not None:
                indices = eval_history.get_all_current_indices()
                history = (eval_history, (all_agent_indices,) + indices)
            else:
                history = None
            _, action, action_log_prob, rnn_states = policy.act(
                obs, rnn_states, masks, all_agent_indices,
                history=history,
                deterministic=False
            )
            if dump_latents:
                assert policy.last_latents is not None
                assert len(policy.last_latents) == num_eval_policies
                for i in range(num_eval_policies):
                    eval_stats_by_opponent['latents'][i][-1].append(policy.last_latents[i].clone())
            policy_pred = policy.aux_pol_cls_head(policy.last_latents) if use_policy_cls_reward else None

        # Obser reward and next obs
        if str(action.device) == 'cpu':
            # This is somehow required by the VecEnv wrapper for action tensors on CPU
            action = action.unsqueeze(-1)

        if args.shuffle_agents:
            prev_agent_perm_all = torch.stack(eval_envs.env_method('get_callback_state', 'agent_shuffler'))
        else:
            prev_agent_perm_all = None

        obs, reward, done, infos = eval_envs.step(action.squeeze(-1))

        if use_policy_cls_reward:
            policy_cls_reward_tracker.advance(reward, infos, policy_pred, prev_agent_perm_all, inspect_idx)
            input(f'Actual reward {reward[inspect_idx]}, policy classification reward {infos[inspect_idx]["expl_reward"]}')

        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                eval_stats_by_opponent['reward'][i].append(info['episode']['r'])
                if args.env_name == 'Overcooked':
                    eval_stats_by_opponent['success_rate'][i].append(
                        int(info['termination_info'].endswith('completed'))
                    )
                    if args.visit_reward_coef is not None:
                        if eval_history is not None:
                            assert ('interaction_stats' in info) == (eval_history.current_episode[i] == args.history_size - 1)
                        if 'interaction_stats' in info:
                            eval_stats_by_opponent['visits_per_interaction'][i].append(
                                info['interaction_stats']['visit_cnt']
                            )
                if args.env_name == 'KuhnPoker':
                    eval_stats_by_opponent['showdown'][i].append(info['showdown'])

            if update_history:
                if args.self_obs_mode:
                    if args.self_action_mode:
                        eval_history.add(i, last_obs[i], action[i], reward[i][0] if eval_history.has_rew_done else None)
                    else:
                        eval_history.add(i, last_obs[i], info['opponent_act'] if 'opponent_act' in info else None,
                                         reward[i][0] if eval_history.has_rew_done else None)
                elif 'opponent_obs' in info:
                    eval_history.add(i, info['opponent_obs'], info['opponent_act'],
                                     reward[i][0] if eval_history.has_rew_done else None)
                if done[i]:
                    if args.self_obs_mode:
                        eval_history.add(i, torch.from_numpy(info['terminal_observation']).float(), None,
                                         0.0 if eval_history.has_rew_done else None)
                    eval_history.finish_episode(i)

        if (args.opponent_switch_period_min is not None or args.opponent_switch_schedule is not None) and update_history:
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    # Reward decreases significantly, clear context
                    metric = 'success_rate' if args.env_name == 'Overcooked' else 'reward'
                    # print(metric)
                    num_full_episodes = eval_history.current_episode[i]
                    reward_max = max(eval_stats_by_opponent[metric][i][-num_full_episodes:])
                    reward_min = min(eval_stats_by_opponent[metric][i][-num_full_episodes:])
                    reward_threshold = reward_max * args.reward_drop_ratio + reward_min * (1.0 - args.reward_drop_ratio)
                    # input()
                    if num_full_episodes > 1 and eval_stats_by_opponent[metric][i][-1] < reward_threshold:
                        # print(
                        #     f'Clearing history for opponent {i} due to reward drop, current total num episodes: {len(eval_stats_by_opponent[metric][i])}, current completed episodes before last clearning {num_full_episodes}'
                        # )
                        # print(f'Process {i} episode {len(eval_stats_by_opponent[metric][i])} {metric}: {eval_stats_by_opponent[metric][i]}, threshold: {reward_threshold}, max: {reward_max}, min: {reward_min}')
                        # input()
                        eval_history.clear_to_one_episode(i)

        last_obs = obs

        if args.use_meta_episode:
            masks = torch.FloatTensor([[0.0] if done_ and len(eval_stats_by_opponent['reward'][i]) % eval_episodes == 0 else [1.0]
                                       for i, done_ in enumerate(done)]).to(obs.device)
        else:
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(obs.device)

    # Mean over episodes, no action over opponents
    mean_eval_stats_by_opponent = {
        k: None if min(len(eval_stats_by_opponent[k][i]) for i in range(num_eval_policies)) < eval_episodes
        else tuple(np.mean(eval_stats_by_opponent[k][i][:eval_episodes]) for i in range(num_eval_policies))
        for k in eval_stats_by_opponent if k != 'latents'
    }
    # Mean over opponents, cumulative mean over episodes
    mean_cumul_eval_stats = {
        k: None if min(len(eval_stats_by_opponent[k][i]) for i in range(num_eval_policies)) < eval_episodes
        else np.cumsum(np.array([eval_stats_by_opponent[k][i][:eval_episodes] for i in range(num_eval_policies)]).mean(axis=0)) / (np.arange(eval_episodes) + 1)
        for k in eval_stats_by_opponent if k != 'latents'
    }
    # Mean over opponents, no action over episodes
    mean_cur_eval_stats = {
        k: None if min(len(eval_stats_by_opponent[k][i]) for i in range(num_eval_policies)) < eval_episodes
        else np.array([eval_stats_by_opponent[k][i][:eval_episodes] for i in range(num_eval_policies)]).mean(axis=0)
        for k in eval_stats_by_opponent if k != 'latents'
    }

    eval_info = {
        'reward': np.mean(mean_eval_stats_by_opponent['reward']),
        'reward_per_opponent': np.array(mean_eval_stats_by_opponent['reward']),
        'cumul_reward': mean_cumul_eval_stats['reward'],
        'cur_reward': mean_cur_eval_stats['reward']
    }

    if dump_latents:
        eval_info.update(
            latents=[eval_stats_by_opponent['latents'][i][:eval_episodes] for i in range(num_eval_policies)]
        )

    if args.env_name == 'Overcooked':
        eval_info.update({
            'success_rate': np.mean(mean_eval_stats_by_opponent['success_rate']),
            'success_rate_per_opponent': np.array(mean_eval_stats_by_opponent['success_rate']),
            'cumul_success_rate': mean_cumul_eval_stats['success_rate'],
            'cur_success_rate': mean_cur_eval_stats['success_rate']
        })
        if args.visit_reward_coef is not None:
            eval_info.update({
                'visits_per_interaction': np.mean([eval_stats_by_opponent['visits_per_interaction'][i][0] for i in range(num_eval_policies)]),
            })

    if args.env_name == 'KuhnPoker':
        eval_info.update({
            'cur_showdown': mean_cur_eval_stats['showdown']
        })

    if log_steps is not None:
        log_group = 'eval' if use_latent else 'eval_ind'
        wandb.log({f'{log_group}/{k}': v for k, v in eval_info.items()}, step=log_steps)

    print(" Evaluation using {} episodes: {}\n".format(num_eval_policies * eval_episodes,
                                                       {k: v for k, v in eval_info.items() if 'cumul' not in k and 'cur' not in k and k != 'latents'}))
    if args.env_name == 'Overcooked':
        print('All success rates, by opponent policy:', mean_eval_stats_by_opponent['success_rate'])

    if update_history:
        eval_history.trim()

    return eval_info
