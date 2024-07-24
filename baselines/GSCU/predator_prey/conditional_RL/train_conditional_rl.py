import os
import argparse
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
import pickle
import logging
import torch
import torch.nn.functional as F
from baselines.GSCU.predator_prey.embedding_learning.opponent_models import *
# from baselines.GSCU.predator_prey.embedding_learning.data_generation import get_all_adv_policies
from baselines.GSCU.predator_prey.conditional_RL.conditional_rl_model import PPO_VAE
from baselines.GSCU.predator_prey.utils.multiple_test import play_multiple_times_train
from baselines.GSCU.predator_prey.utils.config_predator_prey import Config
from learning.envs import make_env
from environment.mpe.policy_both import get_train_eval_pool
import random

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# N_ADV = 3
# adv_pool = Config.ADV_POOL_SEEN
# n_adv_pool = len(adv_pool)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def get_three_adv_all_policies(env, adv_pool):
#     all_policies_idx0 = get_all_adv_policies(env,adv_pool,agent_index=0)
#     all_policies_idx1 = get_all_adv_policies(env,adv_pool,agent_index=1)
#     all_policies_idx2 = get_all_adv_policies(env,adv_pool,agent_index=2)
#     all_policies = [all_policies_idx0,all_policies_idx1,all_policies_idx2]
#     return all_policies

def main(args_):
    from baselines.GSCU.predator_prey.utils.config_predator_prey import args
    print(args_)
    args = Namespace(**vars(args_), **vars(args))
    print(args)
    Transition_vae = namedtuple('Transition_vae', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'latent', 'obs_traj', 'act_traj'])

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gamma = 0.99
    hidden_dim = Config.HIDDEN_DIM
    num_episodes = 300000  # 15 million steps
    actor_lr = args.lr1
    critic_lr = args.lr2
    window_size = Config.WINDOW_SIZW

    checkpoint_freq = 1000
    adv_change_freq = 1
    batch_size = 4000
    ppo_update_freq = 10
    test_freq = 500

    result_dir  = Config.RL_TRAINING_RST_DIR
    os.makedirs(result_dir, exist_ok=True)

    exp_id = args.version
    settings = {}
    settings['exp_id'] = exp_id
    settings['hidden_dim'] = hidden_dim
    settings['actor_lr'] = actor_lr
    settings['critic_lr'] = critic_lr
    settings['batch_size'] = batch_size
    settings['ppo_update_freq'] = ppo_update_freq
    settings['adv_change_freq'] = adv_change_freq
    settings['seed'] = seed

    print(settings)
    env_vae = make_env(args, 'MPE', 0, 0, None, True)()
    train_pool, _ = get_train_eval_pool(args)
    n_adv_pool = len(train_pool)
    # scenario = scenarios.load(args.scenario).Scenario()
    # world_vae = scenario.make_world()
    #
    # env_vae = MultiAgentEnv(world_vae, scenario.reset_world, scenario.reward, scenario.observation,
    #                         info_callback=None, shared_viewer=False, discrete_action=True)

    env_vae.seed(seed)
    env_vae.set_id(args.player_id)
    
    np.random.seed(seed)
    state_dim = env_vae.observation_space.shape[0]
    action_dim = env_vae.action_space.n
    embedding_dim = Config.LATENT_DIM
    encoder_weight_path = Config.VAE_MODEL_DIR + args.encoder_file

    agent_vae = PPO_VAE(state_dim+action_dim, hidden_dim, embedding_dim, action_dim, actor_lr, critic_lr, encoder_weight_path, gamma, n_adv_pool)
    agent_vae.batch_size = batch_size
    agent_vae.ppo_update_time = ppo_update_freq

    return_list = []
    test_return_list = []

    # all_policies_vae = get_three_adv_all_policies(env_vae, adv_pool)
    
    selected_adv_idx = 0
    from tqdm import trange
    for i in trange(50):
        for i_episode in trange(int(num_episodes/50)):
            if i_episode % adv_change_freq == 0:
                # policies_vae = []
                policy_vec = np.zeros(n_adv_pool)
                selected_adv_idx = np.random.randint(0,n_adv_pool)
                policy_vec[selected_adv_idx] += 1
                env_vae.set_opponent(train_pool[selected_adv_idx])
                env_vae.full_reset()
                # for j in range(N_ADV):
                #     policies_vae.append(all_policies_vae[j][selected_adv_idx])
            
            episode_return_vae = 0
            obs = env_vae.reset()
            # obs_n_vae = env_vae._reset()
            policy_vec_tensor = torch.tensor([policy_vec], dtype=torch.float).to(device)

            obs_traj = [np.zeros(state_dim)]*(window_size-1)
            act_traj = [np.zeros(action_dim)] * (window_size)
            hidden = [torch.zeros((1,1,hidden_dim)).to(device), torch.zeros((1,1,hidden_dim)).to(device)]

            for st in range(args.horizon):
                # act_n_vae = []
                #
                # for j, policy in enumerate(policies_vae):
                #     act_vae = policy.action(obs_n_vae[j])
                #     act_n_vae.append(act_vae)
                latent, mu, _ = agent_vae.encoder(policy_vec_tensor)

                obs_traj.append(obs)
                obs_traj_tensor = torch.tensor([obs_traj], dtype=torch.float).to(device)
                act_traj_tensor = torch.tensor([act_traj], dtype=torch.float).to(device)
                act, act_index, act_prob = agent_vae.select_action(obs_traj_tensor, act_traj_tensor, hidden, latent, 0)

                # act_n_vae.append(act)
                # next_obs_n_vae, reward_n_vae, _,_ = env_vae._step(act_n_vae)
                # next_obs_n_vae, reward_n_vae, _,_ = env_vae.step(act)
                # print(act, act_index)
                next_obs, reward, _, _ = env_vae.step(act_index)
                latent = latent[0].cpu().detach().numpy()

                if len(obs_traj) >= window_size:
                    trans_vae = Transition_vae(obs, act_index, act_prob, reward, next_obs, latent, obs_traj.copy(), act_traj.copy())
                    agent_vae.store_transition(trans_vae)
                    obs_traj.pop(0)
                    act_traj.pop(0)

                episode_return_vae += reward
                # episode_return_vae += reward_n_vae[3]
                # obs_n_vae = next_obs_n_vae
                obs = next_obs
                act_traj.append(act)

            return_list.append(episode_return_vae)

            if len(agent_vae.buffer) >= agent_vae.batch_size:
                agent_vae.update(i_episode)
                            
            current_episode = num_episodes / 50 * i + i_episode + 1
            if current_episode % checkpoint_freq == 0:
                agent_vae.save_params(exp_id + '_' + str(current_episode))
            
            if current_episode % test_freq == 0:
                play_episodes = 100
                test_returns = play_multiple_times_train(
                    env_vae, agent_vae, train_pool, 'vae', 'rule', play_episodes=play_episodes
                )
                mean_test_returns = np.mean(test_returns)
                test_return_list.append(mean_test_returns)

                logging.info("Average returns is {0} at the end of epoch {1}".format(mean_test_returns, current_episode))


                result_dict = {}
                result_dict['version'] = exp_id
                result_dict['num_episodes'] = len(return_list)
                result_dict['test_list_vae'] = test_return_list
                result_dict['settings'] = settings

                pickle.dump(result_dict, open(result_dir+'return_' + exp_id + '.p', "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l1', '--lr1', default=5e-4, help='Actor learning rate')
    parser.add_argument('-l2', '--lr2', default=5e-4, help='Critic learning rate')
    # parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    # parser.add_argument('-st', '--steps', default=50, help='Num of steps in a single run')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-v', '--version', required=True)
    parser.add_argument('-e', '--encoder_file', required=True, help='file name of the encoder parameters')
    arg = parser.parse_args()

    main(arg)
