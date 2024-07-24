import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('.')
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
from itertools import count
import copy
import logging
import pickle
import random
from collections import deque
from embedding_learning.opponent_models import EncoderVAE
from utils.utils import get_onehot, sample_fixed_vector#, get_policy_by_vector
from learning.envs import make_vec_envs, make_env

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
#from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# N_ADV = 1

'''
def evaluate_model(envs, policy, num_procs, device):
    eps_cnt = torch.zeros(num_procs)
    rewards = torch.zeros(num_procs,1)
    obs = envs.reset()
    latent = torch.zeros(num_procs, Config.LATENT_DIM).to(device)
    total_eps = 1
    while (eps_cnt < total_eps).any():
        with torch.no_grad():
            action, act_prob  = policy.select_action(obs, latent)
        obs, reward, done, infos = envs.step(action.squeeze(-1))
        still_cnt = (eps_cnt < total_eps)
        rewards[still_cnt] += reward[still_cnt]
        eps_cnt += done
        # print(rewards)
        # print(done)
        # print(eps_cnt)
    ret = (rewards/total_eps).mean().item()
    rewards /= total_eps
    return rewards.mean().item()
'''

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.GSCU_config == "Hard":
        from baselines.GSCU.overcooked.utils.config_overcooked import Config_Hard as Config
    else:
        from baselines.GSCU.overcooked.utils.config_overcooked import Config


    # load rule-based opponent
    # policy parameters of opponents from the seen pool
    sample_p1 = Config.SAMPLE_P1_SEEN
    num_oppo = len(sample_p1)
    assert args.num_processes == num_oppo

    model_dir = os.path.join(Config.RL_MODEL_DIR, f'seed{args.seed}')
    args.log_dir = os.path.join(Config.RL_TRAINING_RST_DIR, f'seed{args.seed}')
    # make train and test envs
    envs = make_vec_envs(args, 'Overcooked', int(args.seed), args.num_processes, args.log_dir, device,
                         allow_early_resets=True, always_use_dummy=False)
    # eval_envs = make_vec_envs(args, 'Overcooked', int(args.seed), args.num_processes, args.log_dir, device,
    #                      allow_early_resets=True, always_use_dummy=False)
    for i in range(args.num_processes):
        # envs.env_method('set_opponent', sample_p1[0], indices=i)
        # eval_envs.env_method('set_opponent', sample_p1[0], indices=i)
        envs.env_method('set_opponent', sample_p1[i%num_oppo], indices=i)
    envs.env_method('set_id', args.player_id)
    # eval_envs.env_method('set_id', args.player_id)

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)

    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    hidden_dim = Config.HIDDEN_DIM
    embedding_dim = Config.LATENT_DIM

    assert envs.observation_space.shape[0] == state_dim
    assert envs.action_space.n == action_dim

    obs_hid_shape = (state_dim+embedding_dim,)
    actor_critic = Policy(
        obs_hid_shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs_hid_shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    
    vae_model_dir = Config.VAE_MODEL_DIR
    encoder_weight_path = vae_model_dir + args.encoder_file
    vae_encoder = EncoderVAE(num_oppo, hidden_dim, embedding_dim).to(device)
    vae_encoder.load_state_dict(torch.load(encoder_weight_path))#,map_location=torch.device('cpu')))
    print('encoder weight loaded')

    vae_vector = [get_onehot(num_oppo,idx) for idx in range(num_oppo)]
    with torch.no_grad():
        policy_vec_tensor = torch.tensor(np.array(vae_vector)).float().to(device)
        latent,mu,_ = vae_encoder(policy_vec_tensor)
        #latent_np = latent.cpu().numpy()

    #latent = torch.zeros(args.num_processes, embedding_dim).to(device)

    obs = envs.reset()
    obs = torch.cat((obs,latent),-1)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    #episode_rewards = deque(maxlen=10)
    eprew_by_policy = tuple(deque() for _ in range(num_oppo))
    epsuc_by_policy = [0] * num_oppo


    #start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    latent = torch.zeros(args.num_processes, embedding_dim).to(device)

    for j in range(num_updates):
        start = time.time()
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action.squeeze(-1))

            #if done in some processes, resample corresponding latent
            for i, d in enumerate(done):
                if d:
                    with torch.no_grad():
                        new_latent,_,_ = vae_encoder(policy_vec_tensor[i].unsqueeze(0))
                        latent[i] = new_latent

            # concatenate latent vector to obs
            obs = torch.cat((obs,latent),-1)

            for i,info in enumerate(infos):
                if 'episode' in info.keys():
                    eprew_by_policy[i % num_oppo].append(info['episode']['r'])
                    #episode_rewards.append(info['episode']['r'])
                    if info['termination_info'].endswith('completed'):
                        epsuc_by_policy[i % num_oppo] += 1


            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1):
            #save_path = os.path.join(model_dir, args.algo)
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError:
                pass

            torch.save(actor_critic, os.path.join(model_dir, f"update{j+1}.pt"))
        
        num_episodes_per_policy = [len(eprew_by_policy[i]) for i in range(num_oppo)]
        if min(num_episodes_per_policy) >= 1:
            episode_result_ready = True
            episode_rewards = [np.mean(eprew_by_policy[i]) for i in range(num_oppo)]
            episode_success = [epsuc_by_policy[i] / num_episodes_per_policy[i] for i in range(num_oppo)]
        else:
            episode_result_ready = False
            episode_rewards = episode_success = None
        
        end = time.time()
        fps = args.num_steps * args.num_processes / (end - start)

        if j % args.log_interval == 0 and episode_result_ready:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        int(fps),
                        sum(num_episodes_per_policy), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            print(f'Mean/median success rate: {np.mean(episode_success):.2f}/{np.median(episode_success):.2f}, '
                    f'min/max success rate: {np.min(episode_success):.2f}/{np.max(episode_success):.2f}\n')
        
        for i in range(num_oppo):
            eprew_by_policy[i].clear()
            epsuc_by_policy[i] = 0


        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     evaluate(actor_critic, obs_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == '__main__':
    main()

                




