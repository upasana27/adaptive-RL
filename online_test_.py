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
        '--exp-name', type=str, default='test', help='experiment name for organizing trajectories'
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
    parser.add_argument(
        '--save-eval-trajectories', action='store_true', help='save detailed trajectories during evaluation'
    )
    parser.add_argument(
        '--eval-trajectory-dir', type=str, default=None, help='directory to save evaluation trajectories (default: logs/{env_name}/{exp_name}/eval_trajectories)'
    )
    parser.add_argument(
        '--use-l1-partners', action='store_true', help='load L1 pretrained partners for L2 evaluation'
    )
    parser.add_argument(
        '--l1-eval-seeds', type=str, default=None, help='comma-separated list of L1 seeds for evaluation (e.g., "5,10,2")'
    )
    parser.add_argument(
        '--use-l1-specialized', action='store_true',
        help='load SPECIALIZED L1 pretrained partners (LO_PB/LO_TC/PB_TC) for L2 evaluation'
    )
    parser.add_argument(
        '--l1-specialized-train', type=str, default=None,
        help='Specialized L1 train pool spec: "TYPE:seed1,seed2;..." (mirrors train_.py)'
    )
    parser.add_argument(
        '--l1-specialized-eval', type=str, default=None,
        help='Specialized L1 eval pool spec: "TYPE:seed1,seed2;..." (mirrors train_.py)'
    )
    parser.add_argument(
        '--latent-training', action='store_true',
        help='mark partners as latent-training policies (needed by specialized pool builder)'
    )
    parser.add_argument(
        '--dump-action-dist', action='store_true',
        help='also dump per-step action probability distributions (Phase 0 / D3)'
    )
    parser.add_argument(
        '--null-opponent', action='store_true',
        help='use a NullPolicy (always-noop) opponent pool (Phase 0 / D1)'
    )
    parser.add_argument(
        '--partner-policy-paths', type=str, default=None,
        help='Phase 0 / D2: comma-separated checkpoint paths for L1 partners. '
             'Each path becomes one entry in the opponent test_pool. Wraps each as '
             'PretrainedPolicy_test on the opposite player slot.'
    )
    parser.add_argument(
        '--feed-partner-history', action='store_true',
        help='2026-10-20: with --partner-policy-paths, fill each L1 partner\'s history as in its own '
             'training (own obs + ego action, kept across the 5-episode meta-episode). Default off = '
             'old behaviour (partner history never written). PACE-style partners only (self-obs-mode).'
    )
    parser.add_argument(
        '--l0-prober', action='store_true',
        help='Replace the L0 partner with a RecipeProber that cycles through revealing each of '
             'the 6 FI recipes in randomized order (fetch ingredient to divider, never plate). '
             'Tests whether a trained partner reacts to PROBES rather than to a committed reveal.'
    )
    parser.add_argument(
        '--l0-prober-dwell', type=int, default=10,
        help='Steps spent revealing each recipe before the prober switches (default 10).'
    )
    parser.add_argument(
        '--l0-truncate-steps', type=int, default=None,
        help='Make the rule-based L0 partner act for only the first N steps of each episode, '
             'then NOOP forever. Sweeps how much partner action the ego needs before it reacts '
             '(an L0 usually makes its first pickup around step 3-4).'
    )
    parser.add_argument(
        '--l0-policy-type', type=str, default='full', choices=['minimum','medium','full'],
        help="L0 event repertoire. 'full' plates+delivers (a COMMITTED reveal). 'minimum' "
             "fetches an ingredient to the divider but never plates it -- a reveal WITHOUT "
             "commitment, i.e. a PROBE. Use it to test whether a partner responds to probing."
    )
    parser.add_argument(
        '--l0-pair-type', type=str, default=None, choices=['R1R2', 'R3R4', 'R5R6', 'ALL', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
        help='6-recipe (FI) rule-based L0 pool selector, mirroring train_.py. Lets an FI L1 be '
             'evaluated against the same L0 partners it trained on. Rule-based partners bypass '
             'the PretrainedPolicy_test path entirely.'
    )
    parser.add_argument(
        '--l0-prefix-tier', type=str, default=None,
        choices=['pickup', 'show', 'chop', 'full'],
        help='Partner-responsiveness diagnostic. Replace the L0 with a scripted leader that '
             'executes only a PREFIX of one recipe and then HOLDS that signal (NOOP) so the '
             "ego's reaction can be observed. 'pickup' = pick the ingredient up and hold it "
             "(fully reversible); 'show' = park it on the shared divider (fully reversible); "
             "'chop' = chop it and park it (stronger signal, NOT reversible); 'full' = a "
             'complete single-recipe L0, the positive control. No tier can plate or deliver, '
             'so no delivery is expected except in the control.'
    )
    parser.add_argument(
        '--l0-prefix-recipe', type=str, default=None,
        choices=['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
        help='Which recipe the prefix probe proposes. Pair it with --l0-pair-type to label the '
             "cell supported (inside the ego's capability set) or unsupported."
    )
    parser.add_argument(
        '--probe-trace-path', type=str, default=None,
        help='Path PREFIX for the per-step JSONL response trace (".pid<N>.jsonl" is appended, '
             'one file per env worker). Records the ego partner-held ingredient, its chopped '
             'flag, and divider contents every step.'
    )
    parser.add_argument(
        '--probe-tag', type=str, default='',
        help='Free-form label copied into every trace row (e.g. R1R2_seed1).'
    )
    parser.add_argument(
        '--horizon-override', type=int, default=None,
        help='override env horizon at eval time (Phase 0 / D2 tight horizons)'
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
        elif args.env_name == 'LBF':
            print(test_pool[inspected_policy].fruit_support_set)
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

        eval_out = evaluate(args, test_pool, args.test_episodes, envs, test_history, policy, None,
                             use_latent=args.latent_training, update_history=True, inspect_idx=inspected_policy)
        # evaluate() returns (eval_info, recipe_pref_by_opponent)
        eval_info = eval_out[0] if isinstance(eval_out, tuple) else eval_out

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
        if getattr(arg, 'partner_policy_paths', None):
            # Phase 0 / D2: bypass pool builders. Build pool directly from explicit paths.
            from environment.overcooked.policy import PretrainedPolicy_test
            partner_paths = [p.strip() for p in arg.partner_policy_paths.split(',') if p.strip()]
            # 2026-10-17: honor the 6-recipe (FI) domain. PretrainedPolicy_test sizes its
            # history buffer from obs_shape, which defaults to the 3-recipe width (73).
            # With an FI partner that yields 73+6=79 fed into a 115+6=121 encoder and
            # crashes with "mat1 and mat2 shapes cannot be multiplied (0x79 and 121x128)".
            # Read recipe_domain from the env config; default keeps FC/PC behavior.
            _partner_obs_shape = None
            _partner_horizon = None
            try:
                import yaml as _yaml
                with open(arg.env_config) as _f:
                    _cfg = _yaml.safe_load(_f)
                    # History buffer must hold a full episode when --feed-partner-history (e.g. FI H=80).
                    _partner_horizon = _cfg.get('horizon')
                    if _cfg.get('recipe_domain') in ('6recipe', '6recipe_x4'):
                        from environment.overcooked.policy import OBS_SHAPE_6RECIPE
                        _partner_obs_shape = OBS_SHAPE_6RECIPE
                        print(f'Partner pool: 6recipe domain -> obs_shape={_partner_obs_shape}')
            except Exception as _e:
                print(f'Partner pool: could not read recipe_domain from {arg.env_config!r} ({_e}); '
                      f'using default obs_shape')
            built_pool = []
            for pth in partner_paths:
                built_pool.append(PretrainedPolicy_test(
                    latent_training=True,
                    history_size=arg.history_size,
                    has_rew_done=arg.has_rew_done,
                    has_meta_time_step=arg.has_meta_time_step,
                    include_current_episode=arg.include_current_episode,
                    merge_encoder_computation=arg.merge_encoder_computation,
                    last_episode_only=arg.last_episode_only,
                    pop_oldest_episode=arg.pop_oldest_episode,
                    self_obs_mode=arg.self_obs_mode,
                    model_path=pth,
                    eval_history=True,
                    agent_id=1 - arg.player_id,
                    feed_history=getattr(arg, 'feed_partner_history', False),
                    **({'max_episode_length': int(_partner_horizon)}
                       if getattr(arg, 'feed_partner_history', False) and _partner_horizon else {}),
                    **({'obs_shape': _partner_obs_shape} if _partner_obs_shape else {}),
                ))
            train_pol = built_pool
            eval_pol = built_pool
            arg.train_pool_size = len(built_pool)
            print(f'Partner-paths pool: {len(built_pool)} L1 partners loaded')
        elif getattr(arg, 'l0_pair_type', None):
            # FI (6-recipe) rule-based L0 pool, mirroring train_.py's L1-training branch.
            # get_train_eval_l0_pool_6recipe reads arg.l0_pair_type and arg.p.
            from environment.overcooked.policy import get_train_eval_l0_pool_6recipe
            print(f'Using 6-recipe L0 pool, pair_type={arg.l0_pair_type}, p={arg.p}')
            train_pol, eval_pol = get_train_eval_l0_pool_6recipe(arg)
            arg.train_pool_size = len(train_pol)
        elif getattr(arg, 'null_opponent', False):
            # Phase 0 / D1: bypass pool builders. Use NullPolicy stand-ins so the eval
            # loop runs without needing the env-specific FCP plumbing.
            from environment.overcooked.policy import NullPolicy
            n = max(1, arg.train_pool_size or 1)
            train_pol = [NullPolicy() for _ in range(n)]
            eval_pol = train_pol
            arg.train_pool_size = n
        elif arg.use_l1_specialized:
            from environment.overcooked.policy import get_train_eval_specialized_L1_pool
            # get_train_eval_specialized_L1_pool reads args.latent_training directly.
            if not hasattr(arg, 'latent_training'):
                arg.latent_training = True
            spec_train = arg.l1_specialized_train or arg.l1_specialized_eval
            spec_eval = arg.l1_specialized_eval or arg.l1_specialized_train
            train_pol, eval_pol = get_train_eval_specialized_L1_pool(arg, spec_train, spec_eval)
        elif arg.use_l1_partners:
            from environment.overcooked.policy import get_train_eval_level_1_pool
            train_pol, eval_pol = get_train_eval_level_1_pool(arg)
        else:
            from environment.overcooked.policy import get_train_eval_pool
            train_pol, eval_pol = get_train_eval_pool(arg)
    elif arg.env_name == 'KuhnPoker':
        from environment.kuhn_poker.policy_new import get_train_eval_pool
        train_pol, eval_pol = get_train_eval_pool(arg)
    elif arg.env_name == 'MPE':
        from environment.mpe.policy_both import get_train_eval_pool
        train_pol, eval_pol = get_train_eval_pool(arg)
    else:
        raise NotImplementedError

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

    # Phase 0 D1/D2: pools were already built upstream; nothing to override here.
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
