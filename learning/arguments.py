import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--pool-seed', type=int, default=1, help='random seed used for pool generation'
    )
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='number of buffer epochs per num_steps of env interaction (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--mini-batch-size', type=int, help='size of mini batch for PPO & embedding training; mutually exclusive with num_mini_batch'
    )
    parser.add_argument(
        '--num-updates', type=int, help='gradient steps per num_steps of env interaction; mutually exclusive with num_epochs'
    )
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='log interval, one log per n env steps (default: 1000)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100000,
        help='save interval, one save per n env steps (default: 100000)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n env steps (default: None)')
    parser.add_argument(
        '--eval-episodes', type=int, help='number of episodes per evaluation'
    )
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--rnn-hidden-dim', type=int, default=0, help='hidden size of RNN'
    )
    parser.add_argument(
        '--rnn-chunk-length', type=int, help='RNN BPTT chunk length'
    )
    parser.add_argument(
        '--share-actor-critic', action='store_true', help='share network between the actor and the critic'
    )
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use-linear-ent-coef-decay',
        type=int,
    )
    parser.add_argument(
        '--disable-advantage-norm',
        action='store_true',
        help='disable advantage normalization'
    )
    parser.add_argument(
        '--disable-clipped-value-loss', action='store_true', help='disable value loss clipping'
    )
    # General arch params
    parser.add_argument(
        '--act-func', type=str, choices=['tanh', 'relu'], default='tanh',
        help='activation function for everything except for the CNN and TF (if any)'
    )
    # General arch params for CNNs, if any
    parser.add_argument(
        '--hidden-channels', type=int, nargs='*', help='hidden channels for every conv layer in each CNN'
    )
    parser.add_argument(
        '--kernel-sizes', type=int, nargs='*', help='kernel sizes for every conv layer in each CNN'
    )
    parser.add_argument(
        '--strides', type=int, nargs='*', help='strides for every conv layer in each CNN'
    )
    parser.add_argument(
        '--paddings', type=int, nargs='*', help='paddings for every conv layer in each CNN'
    )
    # RL arch params
    parser.add_argument(
        '--hidden-dims', type=int, nargs='*', help='hidden layer dimensions for the MLP in actor and critic'
    )
    # Encoder arch params
    parser.add_argument(
        '--pre-hidden-dims', type=int, nargs='*', help='MLP layers before the aggregation function in the encoder'
    )
    parser.add_argument(
        '--encoder-base', type=str, choices=['mlp', 'tf'], help='net arch for the encoder'
    )
    parser.add_argument(
        '--agg-func', type=str, choices=['mean', 'max', 'sum', 'attn'], default='mean',
        help='aggregation function over individual time steps in the encoder'
    )
    parser.add_argument(
        '--act-after-agg', action='store_true', help='perform activation after aggregation instead of before'
    )
    parser.add_argument(
        '--post-hidden-dims', type=int, nargs='*', help='MLP layers after the aggregation function in the encoder'
    )
    # Encoder arch params for TF, if any
    parser.add_argument(
        '--tf-n-layers', type=int
    )
    parser.add_argument(
        '--tf-n-heads', type=int
    )
    parser.add_argument(
        '--tf-hidden-dim', type=int
    )
    parser.add_argument(
        '--tf-ff-dim', type=int
    )
    parser.add_argument(
        '--tf-dropout', type=float, default=0.0
    )
    parser.add_argument(
        '--tf-pos-emb', type=str, choices=['one_hot', 'float', 'none'], default='one_hot',
        help='type of positional embedding in the transformer; float must be used with all_has_all_time_steps'
    )
    parser.add_argument(
        '--tf-chunk-length', type=int  # Deprecated
    )
    # DQN params
    parser.add_argument(
        '--expl-eps', type=float, help='exploration ratio in DQN'
    )
    parser.add_argument(
        '--expl-decay-steps', type=int, help='eps-greedy scheduling steps, 0 to disable'
    )
    parser.add_argument(
        '--expl-eps-final', type=float, help='final eps-greedy exploration ratio'
    )
    parser.add_argument(
        '--buffer-size', type=int, help='buffer size for DQN'
    )
    parser.add_argument(
        '--dueling', action='store_true', help='use dueling DQN'
    )
    parser.add_argument(
        '--target-update-period', type=int, help='period for updating target Q function'
    )
    # Latent training params
    parser.add_argument(
        '--fast-encoder', action='store_true', help='avoid duplicate latent computation during training'
    )
    parser.add_argument(
        '--identity-encoder', action='store_true', help='replace encoder output with policy index'
    )
    parser.add_argument(
        '--emb-encoder', action='store_true', help='replace encoder with an embedding'
    )
    parser.add_argument(
        '--tabular-actor', action='store_true', help='replace latent actor network with a table'
    )
    parser.add_argument(
        '--tabular-critic', action='store_true', help='replace critic networks with tables'
    )
    parser.add_argument(
        '--equal-sampling', action='store_true', help='sample the same amount of data for every policy'
    )
    parser.add_argument(
        '--joint-training', action='store_true', help='train a joint response to all policies'
    )
    parser.add_argument(
        '--latent-training', action='store_true', help='train latent actor'
    )
    parser.add_argument(
        '--use-latent-critic', action='store_true', help='use joint latent critic instead of individual critics'
    )
    parser.add_argument(
        '--alt-training', type=int, help='alternate training of encoder and latent actor'
    )
    parser.add_argument(
        '--collect-all', action='store_true', help='collect all data using individual actors before optimizing'
    )
    parser.add_argument(
        '--recon-obj', action='store_true', help='replace PPO loss with reconstruction loss'
    )
    parser.add_argument(
        '--value-obj', action='store_true', help='train an additional Q function; replace PPO loss with value matching loss'
    )
    parser.add_argument(
        '--value-norm', type=int, default=1, help='norm of value matching loss'
    )
    parser.add_argument(
        '--e2e-obj', action='store_true', help='end-to-end training of latent space'
    )
    parser.add_argument(
        '--soft-imitation-init-prob', type=float,
        help='initial probability (per episode) of using BR policy for rollout'
    )
    parser.add_argument(
        '--soft-imitation-prob-dist', type=str, choices=['const', 'linear'], default='const',
        help='distribution of probability across the interaction episodes'
    )
    parser.add_argument(
        '--soft-imitation-decay-steps', type=int,
        help='number of steps before the probability of using BR policy decays to 0'
    )
    parser.add_argument(
        '--soft-imitation-init-ppo-clip', type=float,
        help='set to enable PPO clip ratio linear scheduling'
    )
    parser.add_argument(
        '--soft-imitation-ratio-clip', type=float,
        help='float >= 1 or None, clip the importance sampling ratio to [1/ratio, ratio], None to disable'
    )
    parser.add_argument(
        '--pcgrad', action='store_true', help='enable PCGrad in latent actor training'
    )
    parser.add_argument(
        '--latent-dim', type=int, help='size of the latent code'
    )
    parser.add_argument(
        '--kl-coef', type=float, help='coefficient of the KL regularization for the latent space'
    )
    parser.add_argument(
        '--kl-cycle', type=int, help='KL scheduling cycle where KL coefficient goes from 0 to kl_coef in kl_cycle steps'
    )
    parser.add_argument(
        '--discrete-latent', action='store_true', help='set discrete latent space'
    )
    parser.add_argument(
        '--quantize-latent', type=int, default=0, help='size of VQ codebook, 0 to use regular VAE'
    )
    parser.add_argument(
        '--deterministic-latent', action='store_true', help='use deterministic latent code'
    )
    parser.add_argument(
        '--vqvae-beta-coef', type=float, default=0, help='coefficient of the commitment loss for VQVAE'
    )
    parser.add_argument(
        '--history-use-episodes', action='store_true',
        help='make history_size, history_full_size and history_refresh_interval in episodes'
    )
    parser.add_argument(
        '--opponent-switch-period-min', type=int
    )
    parser.add_argument(
        '--opponent-switch-period-max', type=int
    )
    parser.add_argument(
        '--history-middle-sampling', action='store_true', help='allow samples ending in the middle of an episode'
    )
    parser.add_argument(
        '--history-full-size', type=int, help='max total number of steps in the history storage'
    )
    parser.add_argument(
        '--history-refresh-interval', type=int, default=1,
        help='interval for refreshing the current history, 1 for always using the latest history'
    )
    parser.add_argument(
        '--history-size', type=int, help='size of history buffer for every opponent policy'
    )
    parser.add_argument(
        '--sample-size', type=int, help='sample size for sampling the history buffer'
    )
    parser.add_argument(
        '--clear-history-on-full', action='store_true', help='clear history when buffer is full'
    )
    parser.add_argument(
        '--separate-history', action='store_true', help='separate history buffer for every process'
    )
    parser.add_argument(
        '--has-rew-done', action='store_true',
        help='whether reward and done info are concatenated to the observation for the history only'
    )
    parser.add_argument(
        '--has-meta-time-step', action='store_true', help='concatenate meta time step to history only'
    )
    parser.add_argument(
        '--all-has-rew-done', action='store_true',
        help='concatenate (last step\'s) reward and done info to the observation'
    )
    parser.add_argument(
        '--all-has-all-time-steps', action='store_true', help='put episode and meta time step into the observation'
    )
    parser.add_argument(
        '--auxiliary-policy-cls-coef', type=float,
        help='enable policy classification as an auxiliary task for the encoder'
    )
    parser.add_argument(
        '--auxiliary-value-pred-coef', type=float,
        help='enable value prediction as an auxiliary task for the encoder'
    )
    parser.add_argument(
        '--contrastive-n-layers', type=int, help='number of layers for the projection head of SimCLR'
    )
    parser.add_argument(
        '--contrastive-tau', type=float, help='temperature for the NT-Xent loss'
    )
    # parser.add_argument(
    #     '--separate-encoder', action='store_true', help='use a separate encoder for the auxiliary task(s)'
    # )
    parser.add_argument(
        '--self-obs-mode', action='store_true', help='put self observation and opponent action (if any) in history'
    )
    parser.add_argument(
        '--self-action-mode', action='store_true', help='put self action instead of opponent action in history, must be used with self_obs_mode'
    )
    parser.add_argument(
        '--last-episode-only', action='store_true', help='only use the last episode in the history'
    )
    parser.add_argument(
        '--pop-oldest-episode', action='store_true', help='pop the oldest episode in history instead of starting a new period'
    )
    parser.add_argument(
        '--auxiliary-transition-pred-coef', type=float, help='predict s\' and r using s, a pair'
    )
    parser.add_argument(
        '--step-mode', action='store_true', help='enable step-based history, with past steps in the current episode; '
                                                 'should be used with self_obs_mode to prevent information leakage'
    )
    parser.add_argument(
        '--include-current-episode', action='store_true',
        help='include information from the current episode in the history under episode mode instead of step mode '
             '(preserve compatibility with the acceleration implementation); '
             'use with self_obs_mode to prevent information leakage'
    )
    parser.add_argument(
        '--encoder-epochs', type=int, help='number of epochs for encoder training'
    )
    parser.add_argument(
        '--encoder-updates', type=int,
        help='number of updates for each encoder training iteration, must be used with encoder_mini_batch_size'
    )
    parser.add_argument(
        '--encoder-mini-batch-size', type=int, help='mini batch size for encoder training'
    )
    parser.add_argument(
        '--encoder-max-samples-per-period', type=int,
        help='maximum # samples per period for encoder training, None for no limit'
    )
    parser.add_argument(
        '--encoder-update-interval', type=int, help='train encoder after every encoder_update_interval steps'
    )
    parser.add_argument(
        '--merge-encoder-computation', action='store_true', help='(maybe) accelerate encoder computation'
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
        '--policy-cls-reward-decay-steps', type=int,
        help='number of steps to decay policy classification reward coefficient to 0'
    )
    parser.add_argument(
        '--policy-cls-warmup-steps', type=int,
        help='number of steps to warmup encoder using only classification without RL'
    )
    parser.add_argument(
        '--ent-coef-decay-steps', type=int, default=0, help='total steps for entropy coefficient decay, 0 to disable'
    )
    parser.add_argument(
        '--contrastive-coef', type=float, default=0.0, help='coefficient of contrastive learning InfoNCE loss'
    )
    parser.add_argument(
        '--pretrained-policy-dir', type=str,
        help='directory of pretrained best responses, set to skip first stage training'
    )
    # Self-play training
    parser.add_argument(
        '--multi-agent', type=int, default=1, help='train a self-play multi-agent policy'
    )
    parser.add_argument(
        '--separate-model', action='store_true', help='train separate models for each agent'
    )
    # Miscellaneous
    parser.add_argument(
        '--train-pool-size', type=int, help='number of training opponents'
    )
    parser.add_argument(
        '--eval-pool-size', type=int, help='number of eval opponents'
    )
    parser.add_argument(
        '--opponent-id', type=int, help='independent RL opponent id, must be used with train_pool_size=1'
    )
    parser.add_argument(
        '--exp-name', type=str, help='name of experiment'
    )
    parser.add_argument(
        '--wandb-user-name', type=str, help='user name for wandb logging, leave empty to disable wandb logging'
    )
    parser.add_argument(
        '--wandb-project-name', type=str, help='project name for wandb logging, leave empty to use the environment name'
    )
    parser.add_argument(
        '--wandb-comment', type=str, default='', help='notes for wandb'
    )
    parser.add_argument(
        '--save-data', action='store_true', help='save collected rollout data, must be used with collect_all'
    )
    parser.add_argument(
        '--load-data-dir', type=str, help='load rollout data from a directory'
    )
    parser.add_argument(
        '--save-partial-ckpt', type=int, help='save suboptimal checkpoints whenever success rate increases by 1/this value'
    )
    parser.add_argument(
        '--use-dummy-vec-env', action='store_true', help='always use dummy vectorized env'
    )
    parser.add_argument(
        '--use-meta-episode', action='store_true', help='use meta-episode that spans multiple episodes'
    )
    parser.add_argument(
        '--pretrained-encoder-dir', type=str, help='directory of pretrained encoder'
    )
    parser.add_argument(
        '--all-has-last-action', action='store_true'
    )
    parser.add_argument(
        '--collect-peer-traj', action='store_true'
    )
    parser.add_argument(
        '--auxiliary-peer-obs-pred-coef', type=float
    )
    parser.add_argument(
        '--auxiliary-peer-act-pred-coef', type=float
    )
    parser.add_argument(
        '--collect-next-obs', action='store_true', help='collect next observation'
    )
    # Overcooked env config
    parser.add_argument(
        '--env_config', type=str, default='./environment/overcooked/config/default.yaml', help='The filepath of the env config yaml file'
    )
    parser.add_argument(
        '--horizon', type=int, help='the horizon of the Overcooked environment, overriding the value in the config file; or the horizon in MPE'
    )
    parser.add_argument(
        '--player-id', type=int, help='the player id of the trained agent'
    )
    parser.add_argument(
        '--p', type=float, default=1.0, help='the probability upper bound for random action'
    )
    parser.add_argument(
        '--desire-id', type=int, help='self-play desire id'
    )
    parser.add_argument(
        '--good-pool-only', action='store_true', help='only use good opponents'
    )
    parser.add_argument(
        '--rule-based-opponents', type=int, default=0, help='number of rule-based opponents'
    )
    parser.add_argument(
        '--recipe-type', type=str, choices=['full', 'cross'], default='full', help='recipe type for Overcooked'
    )
    parser.add_argument(
        '--visit-reward-type', type=str, choices=['step', 'episode', 'interaction'],
        help='visit reward type for Overcooked'
    )
    parser.add_argument(
        '--visit-reward-coef', type=float, help='visit reward coefficient for Overcooked'
    )
    # Kuhn Poker env config
    parser.add_argument(
        '--allow-all-opponents', action='store_true', help='allow dominated opponents in Kuhn Poker'
    )
    # MPE Config
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
        '--separate-patterns', action='store_true', help='separate train and eval prey pattern in MPE'
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr', 'dqn']
    if args.algo == 'ppo':
        assert args.num_epochs is not None and args.num_mini_batch is not None
    elif not args.latent_training:
        assert args.num_updates is not None and args.mini_batch_size is not None
    assert args.recurrent_policy == (args.rnn_hidden_dim > 0)
    assert args.recurrent_policy == (args.rnn_chunk_length is not None)
    assert args.hidden_channels is args.kernel_sizes is args.strides is args.paddings is None \
           or len(args.hidden_channels) == len(args.kernel_sizes) == len(args.strides) == len(args.paddings), \
        f'CNN params must have the same length: channels {args.hidden_channels}, kernels {args.kernel_sizes}, ' \
        f'strides {args.strides}, paddings {args.paddings}'
    if args.identity_encoder:
        args.latent_dim = args.train_pool_size
    args.use_advantage_norm = not args.disable_advantage_norm
    args.use_clipped_value_loss = not args.disable_clipped_value_loss
    assert not (args.discrete_latent and args.quantize_latent), \
        'Can\'t use discrete latent and quantized latent simultaneously'
    assert not (args.identity_encoder and args.emb_encoder)
    assert int(args.recon_obj) + int(args.value_obj) + int(args.e2e_obj) <= 1
    if args.tabular_actor:
        assert args.identity_encoder
    assert (args.algo == 'dqn' and not args.latent_training) == (args.target_update_period is not None)
    if args.e2e_obj:
        assert args.latent_training, 'Must train latent actor since end-to-end training is activated'
        assert args.algo == 'ppo', 'End-to-end training must be used with PPO'
    if args.latent_training and not args.e2e_obj:
        assert args.pretrained_policy_dir is not None, \
            'To train the latent actor in a non-end-to-end manner, individual policies must be provided'
    if args.collect_all:
        assert args.latent_training and (args.value_obj or args.recon_obj)
    if args.joint_training:
        assert not args.latent_training
    if args.multi_agent > 1:
        assert not args.latent_training and not args.joint_training
        assert args.env_name == 'Overcooked'
    assert 0.0 <= args.p <= 1.0
    if args.separate_model:
        assert args.multi_agent > 1
    if args.desire_id is not None:
        assert args.multi_agent > 1
        assert args.train_pool_size == 1

    if args.save_data:
        assert args.collect_all, 'Must collect all data to save it'
        assert args.load_data_dir is None, 'It is recommended that you do not save data that is loaded from the disk'

    if args.load_data_dir is not None:
        assert args.collect_all

    if args.deterministic_latent:
        assert args.latent_training and (not args.discrete_latent) and (args.quantize_latent == 0), \
            'Deterministic latent training is only supported for continuous latent space'

    if args.separate_history:
        assert args.latent_training, 'This is not supported for independent training since no history is involved.'

    if args.use_latent_critic:
        assert args.latent_training, 'Latent critic must be used with latent training'

    if args.auxiliary_value_pred_coef is not None and args.auxiliary_value_pred_coef > 0:
        assert args.latent_training, 'Auxiliary value prediction must be used with latent training'
        assert args.pretrained_policy_dir is not None, 'Pretrained critics must be provided'

    assert not (args.latent_training and args.recurrent_policy), \
        'Latent training with recurrent states is not tested, comment this and proceed with caution'

    # if args.history_size is not None:
    #     assert args.latent_training or args.recurrent_policy, 'If history size is set, either latent training ' \
    #                                                           'or recurrent policy must be used'

    assert not (args.has_rew_done and args.all_has_rew_done), 'has_rew_done and all_has_rew_done cannot be both True'

    if args.auxiliary_policy_cls_coef == float('inf'):
        assert args.encoder_mini_batch_size is not None \
               and (args.encoder_epochs is not None or args.encoder_updates is not None)
        assert not (args.encoder_epochs is not None and args.encoder_updates is not None)
    else:
        assert args.encoder_epochs is None
        if args.auxiliary_policy_cls_coef is not None:
            print(f'Training encoder synchronously with RL using ', end='')
            if args.encoder_mini_batch_size is None:
                print('RL mini batch')
                assert args.encoder_updates is None
            else:
                print(f'separately sampled history mini batch of size {args.encoder_mini_batch_size}')
                if args.encoder_updates is None:
                    print('Updating within every RL update step')
                else:
                    print(f'Updating {args.encoder_updates} steps per RL update iteration of '
                          f'{args.num_epochs} * {args.num_mini_batch} = {args.num_epochs * args.num_mini_batch} steps')
                    assert args.num_epochs * args.num_mini_batch >= args.encoder_updates
        else:
            assert args.encoder_mini_batch_size is None and args.encoder_updates is None

    assert not (args.step_mode and args.include_current_episode)
    if (args.step_mode or args.include_current_episode) and not args.self_obs_mode:
        r = input('WARNING: information from the current episode is included but self_obs_mode is not activated, '
                  'this may cause information leakage. Continue? ')
        if r.lower() not in ['y', 'yes']:
            quit()

    if not args.joint_training:
        assert args.include_current_episode, 'This implementation must be used with include_current_episode'
    assert not args.fast_encoder, 'fast_encoder cannot be used with include_current_episode'
    # Functionality reimplemented now.
    # assert args.auxiliary_policy_cls_coef is None, 'This implementation does not support policy classification'
    assert (args.visit_reward_coef is None) == (args.visit_reward_type is None)
    if args.tf_pos_emb == 'float':
        assert args.all_has_all_time_steps, 'float positional embedding must be used with all_has_all_time_steps'
    if args.tf_pos_emb == 'none':
        assert not args.all_has_all_time_steps, 'tf_pos_emb set to none, but observation contains time step info'
    if args.soft_imitation_init_prob is not None and args.soft_imitation_decay_steps is not None:
        assert args.pretrained_policy_dir is not None
        assert args.soft_imitation_init_ppo_clip is None or args.soft_imitation_init_ppo_clip >= args.clip_param
        assert args.soft_imitation_ratio_clip is None, \
            'Importance sampling ratio should not be clipped when merged into the action loss'
    else:
        assert (args.soft_imitation_init_prob is None and args.soft_imitation_decay_steps is None
                and args.soft_imitation_ratio_clip is None and args.soft_imitation_init_ppo_clip is None)

    return args
