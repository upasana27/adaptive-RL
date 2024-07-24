import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('.')
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
from itertools import count
import copy
import logging
import pickle
import random
from embedding_learning.opponent_models import *
from conditional_RL.conditional_rl_model import PPO_VAE
from utils.config_overcooked import Config
from utils.utils import get_onehot, sample_fixed_vector#, get_policy_by_vector
from learning.envs import make_vec_envs, make_env

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# N_ADV = 1


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


def compute_returns(value_preds, rewards, masks, bad_masks, gamma, gae_lambda):
    #value_preds[-1] = next_value
    num_steps = rewards.shape[0]
    num_procs = rewards.shape[1]
    returns = torch.zeros(num_steps, num_procs, 1)
    gae = 0
    for step in reversed(range(num_steps)):
        #print(rewards[step].device,value_preds[step+1].device,masks[step+1].device)
        delta = rewards[step] + gamma * value_preds[step+1] * masks[step+1] - value_preds[step]
        gae = delta + gamma * gae_lambda * masks[step+1] * gae
        gae = gae * bad_masks[step + 1]
        returns[step] = gae + value_preds[step]
    return returns

def main(args):

    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'returns', 'latent'])

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    version = args.version

    # policy parameters of opponents from the seen pool
    sample_p1 = Config.SAMPLE_P1_SEEN
    assert args.num_processes is None
    args.num_processes = len(sample_p1)

    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    n_adv_pool = Config.NUM_ADV_POOL
    assert n_adv_pool == len(sample_p1)
    embedding_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM

    device = torch.device("cuda:0")

    gamma = 0.99
    gae_lambda = 0.95
    actor_lr = 5e-4 
    critic_lr = 5e-4
    #num_steps = (args.train_steps // n_adv_pool)
    num_updates = (args.train_steps // args.num_steps // args.num_processes)
    checkpoint_freq = num_updates//20
    n_test = 10000
    evaluate_freq = num_updates//1000

    batch_size = 1000
    ppo_update_time = 5
    this_player = args.player_id # controling player 0
    is_sample_emb_in_eisode = False # sample embedding at the beginning of an episode, then fix it.

    vae_model_dir = Config.VAE_MODEL_DIR
    rl_training_rst_dir = Config.RL_TRAINING_RST_DIR
    rl_model_dir = Config.RL_MODEL_DIR
    if not os.path.exists(rl_training_rst_dir):
        os.makedirs(rl_training_rst_dir, exist_ok=False) 
    if not os.path.exists(rl_model_dir):
        os.makedirs(rl_model_dir, exist_ok=False)

    encoder_weight_path = vae_model_dir + args.encoder_file
    agent_VAE = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, n_adv_pool, device, rl_model_dir)
    agent_VAE.batch_size = batch_size
    agent_VAE.ppo_update_time = ppo_update_time


    global_evaluate_return_list = []

    #n_samples = 0
    #for i in range(50):
    # prepare the vec env
    envs = make_vec_envs(args, 'Overcooked', int(args.seed), args.num_processes, args.log_dir, device,
                         allow_early_resets=True, always_use_dummy=False)
    eval_envs = make_vec_envs(args, 'Overcooked', int(args.seed), args.num_processes, args.log_dir, device,
                         allow_early_resets=True, always_use_dummy=False)
    for i in range(args.num_processes):
        envs.env_method('set_opponent', sample_p1[0], indices=i)
        eval_envs.env_method('set_opponent', sample_p1[0], indices=i)
        #envs.env_method('set_opponent', sample_p1[i], indices=i)
    envs.env_method('set_id', args.player_id)
    eval_envs.env_method('set_id', args.player_id)

    obs = envs.reset()
    masks = torch.ones(args.num_processes, 1)
    bad_masks = torch.ones(args.num_processes, 1)
    if args.multi_agent == 1:
        assert obs.shape == (args.num_processes, *envs.observation_space.shape)

    for update in range(num_updates):
        # player = this_player

        # randomly sample opponent from the seen pool
        #rand_int = np.random.randint(len(sample_p1))
        #policy_vec = [0,1/3,0] + sample_p1[rand_int]
        vae_vector = [get_onehot(n_adv_pool,idx) for idx in range(n_adv_pool)]
        #opponent_policy = get_policy_by_vector(policy_vec,is_best_response=False)

        # with torch.no_grad():
        #     policy_vec_tensor = torch.tensor(np.array(vae_vector)).float().to(device)
        #     latent,mu,_ = agent_VAE.encoder(policy_vec_tensor)
        #     #latent_np = latent.cpu().numpy()
        latent = torch.zeros(args.num_processes, embedding_dim).to(device)

        obs_list = torch.zeros(args.num_steps + 1, args.num_processes, *envs.observation_space.shape)
        #print(obs.shape)
        #print(obs_list[0].shape)
        obs_list[0].copy_(obs)
        mask_list = torch.zeros( args.num_steps + 1, args.num_processes, 1)
        bad_mask_list = torch.zeros( args.num_steps + 1, args.num_processes, 1)
        mask_list[0].copy_(masks)
        bad_mask_list[0].copy_(bad_masks)
        action_list = torch.zeros(args.num_steps, args.num_processes, 1)
        action_prob_list = torch.zeros(args.num_steps, args.num_processes, 1)
        reward_list = torch.zeros(args.num_steps, args.num_processes, 1)
        value_preds = torch.zeros(args.num_steps + 1, args.num_processes, 1)
        return_list = torch.zeros(args.num_steps, args.num_processes, 1)
        latent_list = torch.zeros(args.num_steps, args.num_processes, embedding_dim)

        for step in range(args.num_steps):

            # state = game.new_initial_state()
            #obs_list = [obs]
            # act_index_list = []
            # act_prob_list = []
            # reward_list = []
            # value_preds = []
            # masks_list = [masks]
            # bad_masks_list = [bad_masks]
            # latents = []
            #dron_feature_list = []

            action, act_prob = agent_VAE.select_action(obs, latent)
            #action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = envs.step(action.squeeze(-1))
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            mask_list[step+1]= masks
            bad_mask_list[step+1] = bad_masks
            obs_list[step+1] = next_obs
            action_list[step] = action
            action_prob_list[step] = act_prob
            reward_list[step] = rewards
            latent_list[step] = latent
            value_preds[step] = agent_VAE.get_value(obs, latent)
            obs = next_obs

            # if done in some processes, resample corresponding latent
            # for i, done in enumerate(dones):
            #     if done:
            #         with torch.no_grad():
            #             new_latent,_,_ = agent_VAE.encoder(policy_vec_tensor[i].unsqueeze(0))
            #             latent[i] = new_latent
        
        value_preds[-1] = agent_VAE.get_value(obs, latent)


        # store 'state', 'action', 'a_log_prob', 'returns', 'latent'
        # for n in range(num_steps):                 
        #     trans = Transition(obs_list[n], act_index_list[n], act_prob_list[n], return_list[n], latent_np)
        #     agent_VAE.store_transition(trans)  

        # if len(agent_VAE.buffer) >= agent_VAE.batch_size:
        returns = compute_returns(value_preds, reward_list, mask_list, bad_mask_list, gamma, gae_lambda)
        agent_VAE.update(obs_list[:-1], action_list, latent_list, action_prob_list, returns)


        if update % checkpoint_freq == 0:
            agent_VAE.save_params(version +'_'+str(update)) 

        if update % evaluate_freq == 0:
            #evaluate_return = evaluate_training_model(n_test, player, agent_VAE, Config, n_adv_pool,device, seed)
            ret = evaluate_model(eval_envs,agent_VAE,args.num_processes,device)
            #ret = torch.sum(reward_list)/args.num_processes
            logging.info("Average returns is {0} at the end of epoch {1}".format(ret, update))
            global_evaluate_return_list.append(ret)

    result_dict = {}
    result_dict['version'] = version
    result_dict['evaluate_return_list'] = global_evaluate_return_list
    result_dict['evaluate_freq'] = evaluate_freq
    pickle.dump(result_dict, open(rl_training_rst_dir+'return_'+version+'.p', "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-v', '--version', default='v0')
    parser.add_argument('--player-id', type=int)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--env_config', type=str)
    parser.add_argument('--num-processes', type=int)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--multi-agent', type=int, default=1)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--all-has-rew-done', action='store_true')
    parser.add_argument('--recurrent-policy', action='store_true')
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='file name of the encoder parameters')
    arg = parser.parse_args()

    main(arg)

                




