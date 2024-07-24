import argparse
import torch
import numpy as np
from learning.envs import make_vec_envs
from learning.storage_ import PeriodicHistoryStorage
from evaluation_ import evaluate
import os
from learning.model import LatentPolicy
from learning.utils import AgentIndicesMapper
from environment.policy_common import DynamicPolicy, MultiAgentResamplePolicy
import pickle


def get_args():
    parser = argparse.ArgumentParser(description='Online test')

    parser.add_argument(
        '--seed', type=int, default=1
    )
    parser.add_argument(
        '--no-cuda', action='store_true'
    )
    parser.add_argument(
        '--cuda-deterministic', action='store_true'
    )
    parser.add_argument(
        '--policy-dir', type=str
    )
    parser.add_argument(
        '--log-dir', type=str, required=True
    )
    parser.add_argument(
        '--env-name', type=str
    )
    # parser.add_argument(
    #     '--opponents', type=str
    # )
    parser.add_argument(
        '--history-size', type=int
    )
    parser.add_argument(
        '--test-pass', type=int, default=1
    )
    parser.add_argument(
        '--interaction-steps', type=int
    )
    parser.add_argument(
        '--test-episodes', type=int
    )
    parser.add_argument(
        '--algo', type=str, default='ppo'
    )
    parser.add_argument(
        '--value-obj', action='store_true'
    )
    parser.add_argument(
        '--joint-training', action='store_true'
    )
    parser.add_argument(
        '--env_config', type=str
    )
    parser.add_argument(
        '--player-id', type=int
    )
    parser.add_argument(
        '--multi-agent', type=int, default=1
    )
    parser.add_argument(
        '--all-has-rew-done', action='store_true'
    )
    parser.add_argument(
        '--recurrent-policy', action='store_true'
    )
    parser.add_argument(
        '--separate-model', action='store_true'
    )
    parser.add_argument(
        '--separate-history', action='store_true'
    )
    parser.add_argument(
        '--has-rew-done', action='store_true'
    )
    parser.add_argument(
        '--self-obs-mode', action='store_true'
    )
    parser.add_argument(
        '--self-action-mode', action='store_true'
    )
    parser.add_argument(
        '--merge-encoder-computation', action='store_true'
    )
    parser.add_argument(
        '--opponent-switch-period-min', type=int
    )
    parser.add_argument(
        '--opponent-switch-period-max', type=int
    )
    parser.add_argument(
        '--opponent-switch-schedule', type=int, nargs='+'
    )
    parser.add_argument(
        '--train-pool-size', type=int,
    )
    parser.add_argument(
        '--eval-pool-size', type=int,
    )
    parser.add_argument(
        '--opponent-id', type=int,
    )
    parser.add_argument(
        '--rule-based-opponents', type=int
    )
    parser.add_argument(
        '--use-meta-episode', action='store_true'
    )
    parser.add_argument(
        '--has-meta-time-step', action='store_true'
    )
    parser.add_argument(
        '--all-has-last-action', action='store_true'
    )
    parser.add_argument(
        '--all-has-all-time-steps', action='store_true'
    )
    parser.add_argument(
        '--include-current-episode', action='store_true'
    )
    parser.add_argument(
        '--desire-id', type=int
    )
    parser.add_argument(
        '--p', type=float, default=1.0
    )
    parser.add_argument(
        '--recipe-type', type=str, choices=['full', 'cross'], default='full', help='recipe type for Overcooked'
    )
    parser.add_argument(
        '--inspected-policy', type=int
    )
    parser.add_argument(
        '--visit-reward-coef', type=float
    )
    parser.add_argument(
        '--visit-reward-mode', type=str
    )
    parser.add_argument(
        '--visit-reward-type', type=str
    )
    parser.add_argument(
        '--pool-seed', type=int, default=1
    )
    parser.add_argument(
        '--last-episode-only', action='store_true'
    )
    parser.add_argument(
        '--pop-oldest-episode', action='store_true',
        help='pop the oldest episode in history instead of starting a new period'
    )
    parser.add_argument(
        '--policy-cls-reward-coef', type=float,
        help='use policy classification result for reward, set to inf to multiply this to the raw reward'
    )
    parser.add_argument(
        '--policy-cls-reward-type', type=str, default='accuracy', choices=['accuracy', 'entropy'],
        help='use accuracy or (scaled) entropy as policy classification reward'
    )
    parser.add_argument(
        '--policy-cls-reward-mode', type=str, default='diff', choices=['diff', 'full', 'max_diff', 'max_full'],
        help='compute difference between two values or the full value as policy classification reward'
    )
    parser.add_argument(
        '--dump-latents', action='store_true'
    )
    parser.add_argument(
        '--scenario', type=str, choices=['simple_tag_multi_partial'], help='MPE scenario, PP only for now'
    )
    parser.add_argument(
        '--num-agents', type=int, help='total number of agents in MPE'
    )
    parser.add_argument(
        '--num-good-agents', type=int, help='number of preys in MPE'
    )
    parser.add_argument(
        '--obs-radius', type=float, help='observation radius in MPE'
    )
    parser.add_argument(
        '--init-radius', type=float, default=1.0, help='agent initial position radius in MPE'
    )
    parser.add_argument(
        '--shaped-reward', action='store_true', help='add shaped reward in MPE'
    )
    parser.add_argument(
        '--collide-reward', action='store_true', help='add collide reward in MPE'
    )
    parser.add_argument(
        '--collide-reward-once', action='store_true', help='each prey can be caught only once'
    )
    parser.add_argument(
        '--watch-tower', action='store_true', help='add watch tower in MPE'
    )
    parser.add_argument(
        '--shuffle-agents', action='store_true', help='shuffle other agents in observation, applicable only to MPE'
    )
    parser.add_argument(
        '--horizon', type=int
    )
    parser.add_argument(
        '--use-dummy-vec-env', action='store_true'
    )
    parser.add_argument(
        '--use-train-pool', action='store_true'
    )
    parser.add_argument(
        '--separate-patterns', action='store_true', help='separate train and eval prey pattern in MPE'
    )
    parser.add_argument(
        '--reward-drop-ratio', type=float, help='ratio of reward drop before clearing context'
    )
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def online_test(args, policy: LatentPolicy, test_pool):
    # print('Testing model', policy)
    torch.set_printoptions(sci_mode=False)
    num_test_policies = len(test_pool)
    if args.opponent_switch_period_min is not None or args.opponent_switch_schedule is not None:
        from copy import deepcopy
        print('Switching opponents; make sure in eval mode')
        # print(test_pool)
        if args.env_name != 'MPE':
            test_pool = [MultiAgentResamplePolicy([1 - args.player_id], None, [deepcopy(test_pool[i])], [i], [len(test_pool)])
                         for i in range(len(test_pool))]
        # print(test_pool)
        # input()
        if args.opponent_switch_schedule is not None:
            assert args.opponent_switch_period_min is None
            assert args.opponent_switch_period_max is None
            assert sum(args.opponent_switch_schedule) == args.test_episodes
            test_pool_ = [DynamicPolicy([deepcopy(test_pool[(i + j) % len(test_pool)]) for j in range(len(args.opponent_switch_schedule))],
                                        None, None, args.opponent_switch_schedule)
                          for i in range(len(test_pool))]
        else:
            assert args.opponent_switch_schedule is None
            test_pool_ = [DynamicPolicy([deepcopy(test_pool[i]), deepcopy(test_pool[(i + 1) % len(test_pool)])],
                                        args.opponent_switch_period_min, args.opponent_switch_period_max)
                          for i in range(len(test_pool))]
        test_pool = test_pool_
        # print(test_pool)
        # input()
        # for p in test_pool:
        #     p.resample()
    args.latent_training = policy.latent_training_mode
    print('Latent training:', args.latent_training)
    if not args.latent_training:
        args.num_trained_policies = len(policy.actors)
    inspected_policy = args.inspected_policy
    if inspected_policy is not None:
        print('Inspecting policy', inspected_policy, end=', ')
        if args.env_name == 'Overcooked':
            print(test_pool[inspected_policy].ingredient_support_set)
        else:
            print(test_pool[inspected_policy].current_policies)
    args.history_use_episodes = True
    device = 'cuda' if args.cuda else 'cpu'
    envs = make_vec_envs(args, args.env_name, args.seed, num_test_policies, args.log_dir, device,
                         allow_early_resets=True, render_rank=inspected_policy, always_use_dummy=args.use_dummy_vec_env)
    for i in range(num_test_policies):
        envs.env_method('set_opponent', test_pool[i], indices=i)
    if args.env_name == 'Overcooked' or args.env_name == 'MPE':
        envs.env_method('set_id', args.player_id)
    max_episode_length = envs.env_method('episode_length', indices=0)[0]
    indices_mapper = AgentIndicesMapper(args)

    if not hasattr(policy, 'share_actor_critic'):
        policy.share_actor_critic = False
    policy.indices_mapper = indices_mapper
    if policy.actors is not None:
        for actor in policy.actors:
            if not hasattr(actor, 'rnn'):
                actor.rnn = None
    if hasattr(policy, 'actor') and policy.actor is not None and not hasattr(policy.actor, 'rnn'):
        policy.actor.rnn = None
    if policy.critics is not None:
        for critic in policy.critics:
            if not hasattr(critic, 'rnn'):
                critic.rnn = None
    if hasattr(policy, 'critic') and policy.critic is not None and not hasattr(policy.critic, 'rnn'):
        policy.critic.rnn = None
    if policy.latent_training_mode:
        policy.actors = None
        policy.critics = None
    else:
        policy.actor = None
        policy.critic = None

    if args.latent_training:
        test_history = PeriodicHistoryStorage(
            num_processes=num_test_policies,
            num_policies=num_test_policies,
            history_storage_size=args.history_size,
            clear_period=args.history_size,
            refresh_interval=1,
            sample_size=None,
            has_rew_done=args.has_rew_done,
            max_samples_per_period=None,
            step_mode=False,
            use_episodes=True,
            has_meta_time_step=args.has_meta_time_step,
            include_current_episode=args.include_current_episode,
            obs_shape=envs.observation_space.shape,
            act_shape=tuple(),
            max_episode_length=max_episode_length,
            merge_encoder_computation=args.merge_encoder_computation,
            last_episode_only=args.last_episode_only,
            pop_oldest_episode=args.pop_oldest_episode,
        )
        test_history.to(device)
    else:
        test_history = None

    all_eval_info = {}

    for ps in range(args.test_pass):
        if test_history is not None:
            test_history.clear()

        print(f'Evaluating #{ps}...')

        # if args.opponent_switch_period_min is not None:
        #     # Start every test pass with a new opponent
        #     for pol in test_pool:
        #         pol.resample()

        eval_info = evaluate(args, test_pool, args.test_episodes, envs, test_history, policy, None,
                             use_latent=args.latent_training, update_history=True, inspect_idx=inspected_policy)

        for k in eval_info:
            if k not in all_eval_info:
                all_eval_info[k] = []
            all_eval_info[k].append(eval_info[k])

        eval_info = {k: (np.mean(v, axis=0), np.std(v, axis=0), len(v)) for k, v in all_eval_info.items() if k != 'latents'}
        print(f'Test pass #{ps}, mean {eval_info}')

    # print(all_eval_info)
    eval_info = {k: (np.mean(v, axis=0), np.std(v, axis=0), len(v)) for k, v in all_eval_info.items() if k != 'latents'}
    print(f'Final result for {args.policy_dir}: {eval_info}')
    save_path = args.policy_dir.rstrip('.pt') + '_all_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(all_eval_info, f)
    # save_path = os.path.dirname(args.policy_dir)
    # np.save(os.path.join(save_path, 'all_results.npy'), all_eval_info)
    # np.save(os.path.join(save_path, 'results.npy'), eval_info)
    import sys
    print('Results saved to', save_path, file=sys.stderr)


if __name__ == '__main__':
    arg = get_args()

    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    np.random.seed(arg.seed)
    device = 'cuda' if arg.cuda else 'cpu'

    if arg.cuda and torch.cuda.is_available() and arg.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if arg.env_name == 'Overcooked':
        from environment.overcooked.policy import get_train_eval_pool
    elif arg.env_name == 'KuhnPoker':
        from environment.kuhn_poker.policy_new import get_train_eval_pool
    elif arg.env_name == 'MPE':
        from environment.mpe.policy_both import get_train_eval_pool
    else:
        raise NotImplementedError
    train_pol, eval_pol = get_train_eval_pool(arg)

    # Prepare and check peer ids
    if arg.env_name != 'MPE':
        assert arg.num_agents is None
        arg.num_agents = 2
        arg.policy_id_max = torch.tensor([arg.train_pool_size], dtype=torch.long)
        arg.policy_id_all = torch.arange(arg.train_pool_size).unsqueeze(0)
    else:
        assert all(pol.max_ids == train_pol[0].max_ids for pol in train_pol)
        assert all(pol.max_ids == train_pol[0].max_ids for pol in eval_pol)
        assert arg.num_agents - arg.num_good_agents > 1, 'There must be at least 1 peer predator present'
        if arg.shuffle_agents:
            # Every agent could be predator or prey. Merge their IDs
            arg.policy_id_max = torch.full((arg.num_agents - 1,),
                                           train_pol[0].max_ids[0] + train_pol[0].max_ids[-1],
                                           dtype=torch.long)
            arg.policy_id_all = torch.tensor([pol.current_ids for pol in train_pol]).T
            # Predator IDs precede prey IDs. Add the offset to prey IDs
            arg.policy_id_all[-arg.num_good_agents:] += train_pol[0].max_ids[0]
        else:
            arg.policy_id_max = torch.tensor(train_pol[0].max_ids)
            arg.policy_id_all = torch.tensor([pol.current_ids for pol in train_pol]).T
    assert arg.policy_id_max.shape == (arg.num_agents - 1,)
    assert arg.policy_id_all.shape == (arg.num_agents - 1, arg.train_pool_size)
    assert (arg.policy_id_all < arg.policy_id_max.unsqueeze(-1)).all()

    if arg.opponent_id is not None:
        test_pol = [train_pol[arg.opponent_id]]
    elif arg.use_train_pool:
        test_pol = train_pol
    else:
        test_pol = eval_pol
    # test_pol = eval_pol
    # test_pol = [train_pol[1]]

    # test_pol = get_test_pool(arg)[0][:25]
    # print('Test policies:', [pol.model_path for pol in test_pol])
    # if arg.opponent_switch_period_min is not None:
    #     # Construct the same number of test policies
    #     # assert arg.test_episodes % arg.opponent_switch_period == 0
    #     print('Using dynamic opponent, switch period between', arg.opponent_switch_period_min,
    #           'and', arg.opponent_switch_period_max, 'episodes')
    #     test_pol = [DynamicPolicy(test_pol, arg.opponent_switch_period_min, arg.opponent_switch_period_max)
    #                 for _ in test_pol]

    if '%OPP_ID%' in arg.policy_dir:
        model = torch.load(arg.policy_dir.replace('%OPP_ID%', '0'), map_location=device)
        assert len(model.actors) == len(model.critics) == 1
        i = 1
        while os.path.exists(arg.policy_dir.replace('%OPP_ID%', str(i))):
            model_ = torch.load(arg.policy_dir.replace('%OPP_ID%', str(i)), map_location=device)
            assert len(model_.actors) == len(model_.critics) == 1
            model.actors.extend(model_.actors)
            model.critics.extend(model_.critics)
            i += 1
        print('Loaded', i, 'models')
        assert len(model.actors) == len(test_pol), f'Number of models does not match number of test policies, {len(test_pol)}'
    else:
        model = torch.load(arg.policy_dir, map_location=device)

    print(f'Testing policy {arg.policy_dir} against {len(test_pol)} test policies')
    if arg.env_name == 'KuhnPoker':
        print('Theoretical optimum:', np.mean([p.get_best_response_return() for p in test_pol]))

    online_test(arg, model, test_pol)
