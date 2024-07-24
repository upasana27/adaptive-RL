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
import pickle
import logging
import random
import pandas as pd 
import glob 
from embedding_learning.opponent_models import *
from bayesian_update import VariationalInference,EXP3
from conditional_RL.conditional_rl_model import PPO_VAE
#from utils.mypolicy import PolicyKuhn,get_policy_by_vector,BestResponseKuhn
from utils.utils import get_p1_region,get_onehot,kl_by_mean_sigma,mse
from learning.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy

np.set_printoptions(precision=6)
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def moving_avg(x,N=5):
    return np.convolve(x, np.ones(N)/N, mode='same')

def region2index(region):
    if region == 7:
        return 0
    elif region == 3:
        return 1
    elif region == 5:
        return 2 
    else:
        return -1

def evaluate_exp3_det(n_test, player, agent_vae, opponent_policy, opponent_type, latent, ne_response, exp3):
    vae_probs = []
    for i in range(6):
        obs = [0.0] * 11
        obs[0] = 1.0
        obs[2 + (i % 3)] = 1.0
        if i >= 3:
            obs[5] = obs[8] = 1.0

        _, action, action_prob = agent_vae.select_action(obs.copy(), latent)
        if action == 0:
            action_prob = 1.0 - action_prob
        assert 0.0 <= action_prob <= 1.0
        vae_probs.append(1.0 if action_prob >= 0.5 else 0.0)
    vae_return = opponent_policy.get_return_complex(vae_probs)
    ne_return = opponent_policy.get_return(ne_response.alpha, ne_response.beta, ne_response.gamma)
    exp3.generate_p()
    return vae_return * exp3.p[0] + ne_return * (1.0 - exp3.p[0]), None

def evaluate_exp3(n_test, player, agent_vae, opponent_policy, opponent_type, latent, ne_response, exp3):
    vae_probs = []
    for i in range(6):
        obs = [0.0] * 11
        obs[0] = 1.0
        obs[2 + (i % 3)] = 1.0
        if i >= 3:
            obs[5] = obs[8] = 1.0

        _, action, action_prob = agent_vae.select_action(obs.copy(), latent)
        if action == 0:
            action_prob = 1.0 - action_prob
        assert 0.0 <= action_prob <= 1.0
        vae_probs.append(action_prob)
    vae_return = opponent_policy.get_return_complex(vae_probs)
    ne_return = opponent_policy.get_return(ne_response.alpha, ne_response.beta, ne_response.gamma)
    exp3.generate_p()
    return vae_return * exp3.p[0] + ne_return * (1.0 - exp3.p[0]), None
    game = pyspiel.load_game("kuhn_poker(players=2)") 
    return_list = []
    for _ in range(n_test):
        state = game.new_initial_state()
        agent_selected = exp3.sample_action()
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            cur_player = state.current_player()
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                s = state.information_state_tensor(cur_player)
                if cur_player == player:
                    if agent_selected == 0:
                        _, action, _ = agent_vae.select_action(s, latent)
                    else:
                        action = ne_response.action(s)
                else:
                    if opponent_type == 'rl':
                        _, action, _ = opponent_policy.select_action(s)
                    else:
                        action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list

def evaluate_vae_det(n_test, player, agent_vae, opponent_policy, opponent_type, latent):
    vae_probs = []
    for i in range(6):
        obs = [0.0] * 11
        obs[0] = 1.0
        obs[2 + (i % 3)] = 1.0
        if i >= 3:
            obs[5] = obs[8] = 1.0

        _, action, action_prob = agent_vae.select_action(obs.copy(), latent)
        if action == 0:
            action_prob = 1.0 - action_prob
        assert 0.0 <= action_prob <= 1.0
        vae_probs.append(1.0 if action_prob >= 0.5 else 0.0)
    vae_return = opponent_policy.get_return_complex(vae_probs)
    return vae_return, None

def evaluate_vae(n_test, player, agent_vae, opponent_policy, opponent_type, latent):
    vae_probs = []
    for i in range(6):
        obs = [0.0] * 11
        obs[0] = 1.0
        obs[2 + (i % 3)] = 1.0
        if i >= 3:
            obs[5] = obs[8] = 1.0

        _, action, action_prob = agent_vae.select_action(obs.copy(), latent)
        if action == 0:
            action_prob = 1.0 - action_prob
        assert 0.0 <= action_prob <= 1.0
        vae_probs.append(action_prob)
    vae_return = opponent_policy.get_return_complex(vae_probs)
    return vae_return, None
    game = pyspiel.load_game("kuhn_poker(players=2)")
    return_list = []
    for _ in range(n_test):
        state = game.new_initial_state()
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            cur_player = state.current_player()
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                s = state.information_state_tensor(cur_player)
                if cur_player == player:
                    _, action, _ = agent_vae.select_action(s, latent)
                else:
                    if opponent_type == 'rl':
                        _, action, _ = opponent_policy.select_action(s)
                    else:
                        action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list

def evaluate_baseline(n_test, player, response, opponent_policy, opponent_type):
    ne_return = opponent_policy.get_return(response.alpha, response.beta, response.gamma)
    return ne_return, None
    game = pyspiel.load_game("kuhn_poker(players=2)")
    return_list = []
    for _ in range(n_test):
        state = game.new_initial_state()
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            cur_player = state.current_player()
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                s = state.information_state_tensor(cur_player)
                if cur_player == player:
                    action = response.action(s)
                else:
                    if opponent_type == 'rl':
                        _, action, _ = opponent_policy.select_action(s)
                    else:
                        action = opponent_policy.action(s)
                state.apply_action(action)
        returns = state.returns()
        this_returns = returns[player]
        return_list.append(this_returns)
    return np.mean(return_list),return_list


def main(args):
    if args.GSCU_config == "Hard":
        from baselines.GSCU.overcooked.utils.config_overcooked import Config_Hard as Config
    elif args.GSCU_config == "Normal":
        from baselines.GSCU.overcooked.utils.config_overcooked import Config
    else:
        raise NotImplementedError
    
    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    n_adv_pool = Config.NUM_ADV_POOL
    embedding_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM
    rst_dir = Config.ONLINE_TEST_RST_DIR
    data_dir = Config.DATA_DIR
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=False) 

    n_steps = 100 # vi update freq, steps
    test_eps = 10 # history context length is 10 episode
    n_test = 15 # number of evaluation context per opponent, each context has 10 episodes
    # this_player = 0 # controlling player

    # n_opponent = 200 # total number of opponent switch (we tested 10 sequences with 20 opponents each)
    # reset_freq = 20 # sequence length
    # n_episode = 1000 # number of episodes per opponent switch
    # evaluation_freq = n_steps * 15 # evaluation freq
    # n_test = 200 # number of evaluation epsideos. More than 1000 episodes are recommanded.

    # n_opponent_real = 700
    # reset_freq = 7
    # n_episode = 1000
    # evaluation_freq = n_episode - 1
    # n_test = 1000

    version = args.version 
    # opponent_type = args.opp_type
    # print ('opponent_type',opponent_type)

    seed = 6#int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cpu")

    encoder_weight_path = Config.VAE_MODEL_DIR + args.encoder_file
    decoder_weight_path = Config.VAE_MODEL_DIR + args.decoder_file
    conditional_rl_weight_path = os.path.join(Config.RL_MODEL_DIR, f'seed{args.seed}', args.rl_file)

    # load rule-based policies
    test_oppo = Config.TEST_OPPO
    num_oppo = len(test_oppo)
    # load conservative policy
    conservative = Config.ONLINE_CONSERVATIVE_POLICY
    conservative.reset()
    assert args.num_processes == num_oppo
    # create vec envs
    envs = make_vec_envs(args, 'Overcooked', int(args.seed), num_oppo, rst_dir, device,
                         allow_early_resets=True, always_use_dummy=False)
    for i in range(args.num_processes):
        envs.env_method('set_opponent', test_oppo[i], indices=i)
        envs.env_method('set_id', 0, indices=i)  #self controlled agent is always player 0
    
    assert envs.observation_space.shape[0] == state_dim
    assert envs.action_space.n == action_dim

    obs_hid_shape = (state_dim+embedding_dim,)

    # load trained vae and RL models
    opponent_model = OpponentModel(state_dim, n_adv_pool, hidden_dim, embedding_dim, action_dim, encoder_weight_path, decoder_weight_path)
    # cond_policy = Policy(
    #     obs_hid_shape,
    #     envs.action_space,
    #     base_kwargs={'recurrent': False})
    cond_policy = torch.load(conditional_rl_weight_path, map_location=device)
    vis = [VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=20, game_steps=n_steps) for _ in range(num_oppo)]

    exp3s = [EXP3(n_action=2, gamma=0.3, min_reward=-10, max_reward=15) for _ in range(num_oppo)] # lr of exp3 is set to 0.3
    #agent_vae = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, 0.0, 0.0, encoder_weight_path, n_adv_pool)
    #agent_vae.init_from_save(conditional_rl_weight_path)
            
    # # a randomly generated sequence is used. You can create your own.
    # policy_vectors_df = pd.read_pickle(data_dir+'online_test_policy_vectors_demo.p')
    # if opponent_type == 'seen':
    #     policy_vectors = policy_vectors_df['seen']
    # elif opponent_type == 'unseen':
    #     policy_vectors = policy_vectors_df['unseen']
    # elif opponent_type == 'mix':
    #     policy_vectors = policy_vectors_df['mix']
    # else:
    #     raise ValueError('No such opponent type')

    # if opponent_type == 'seen':
    #     policy_vectors = [[[0., 1. / 3., 0., p[0], p[1]], None] for p in Config.SAMPLE_P1_SEEN(seed)]
    # elif opponent_type == 'unseen':
    #     policy_vectors = [[[0., 1. / 3., 0., p[0], p[1]], None] for p in Config.SAMPLE_P1_UNSEEN]
    # else:
    #     raise NotImplementedError

    # n_pass = n_opponent_real // len(policy_vectors)
    # n_opponent = len(policy_vectors) * n_pass
    # print(f'After adjustment: {len(policy_vectors)} {opponent_type} policies * {n_pass} passes = {n_opponent} opponents')
    # policy_vectors = policy_vectors * n_pass

    # game = pyspiel.load_game("kuhn_poker(players=2)")
    # state = game.new_initial_state()
    obs = envs.reset()
    cur_eps_cnt = np.zeros(args.num_processes)
    cur_return = np.zeros(args.num_processes)
    cur_obs_list = [[] for _ in range(args.num_processes)]
    cur_act_index_list = [[] for _ in range(args.num_processes)]
    test_cnt = np.zeros(args.num_processes)
    return_list = [[0] for _ in range(args.num_processes)]
    success_cnt = [[0] for _ in range(args.num_processes)]
    success_rates = np.zeros((args.num_processes, n_test, test_eps))
    for exp3 in exp3s:
        exp3.init_weight()
    for vi in vis:
        vi.init_all()
    selected_policy = [exp3.sample_action() for exp3 in exp3s]
    # selected_policy = [1 for _ in range(num_oppo)]
    latent = [vi.generate_cur_embedding(is_np=False) for vi in vis]
    while np.min(test_cnt)<n_test:
        action_env = torch.zeros(args.num_processes, dtype=torch.int32)
        for i in range(num_oppo):
            if selected_policy[i] == 0:
                # select latent conditioned policy
                _,action_env[i],_,_ = cond_policy.act(torch.cat([obs[i], latent[i].squeeze()], dim=-1),None,None)
            else:
                # select conservative policy
                action_env[i] = conservative(obs[i])
        obs, rewards, dones, infos = envs.step(action_env)

        # append opponent obs and action
        for i,info in enumerate(infos):
            if 'opponent_obs' in info or 'opponent_act' in info:
                cur_obs_list[i].append(info['opponent_obs'].numpy())
                cur_act_index_list[i].append(info['opponent_act'].numpy())
            if 'episode' in info.keys():
                if info['termination_info'].endswith('completed'):
                    success_cnt[i][-1] += 1
            if len(cur_obs_list[i])>=n_steps:
                # update vae
                act_index = np.array(cur_act_index_list[i]).astype(np.float32)
                obs_adv_tensor = torch.FloatTensor(np.array(cur_obs_list[i]))

                act_adv_tensor = torch.FloatTensor(np.array(act_index))
                vis[i].update(obs_adv_tensor, act_adv_tensor)
                latent[i] = vis[i].generate_cur_embedding(is_np=False)
                cur_obs_list[i] = []
                cur_act_index_list[i] = []

        for i,done in enumerate(dones):
            cur_return[i] += rewards[i].item()
            if done:
                # update eps cnt
                cur_eps_cnt[i] += 1
                # update success rate
                # if i==1:
                #     print(f'oppo {i} before,{selected_policy[i]}')
                #     print(cur_return[i])

                if len(success_cnt[i])<=n_test:
                    test_pass_idx = len(success_cnt[i])-1
                    cur_eps_num = int(cur_eps_cnt[i])
                    success_rates[i][test_pass_idx][cur_eps_num-1] = success_cnt[i][-1] / cur_eps_num
                # update EXP3
                exp3s[i].update(cur_return[i], selected_policy[i])
                # update return
                return_list[i][-1] += cur_return[i].item()
                cur_return[i] = 0
                # resample policy
                selected_policy[i] = exp3s[i].sample_action()
                # if i==1:
                #     print(f'oppo {i} after,{selected_policy[i]}')
                #     print(success_rates[i])
        
            # if exceed test_eps, reset all i
            if cur_eps_cnt[i] >= test_eps:
                test_cnt[i]+=1
                cur_eps_cnt[i] = 0
                cur_return[i] = 0
                cur_obs_list[i] = []
                cur_act_index_list[i] = []
                return_list[i].append(0)
                #success_rate[i][-1] /= test_eps
                success_cnt[i].append(0)
                exp3s[i].init_weight()
                vis[i].init_all()
                selected_policy[i] = exp3s[i].sample_action()
                latent[i] = vis[i].generate_cur_embedding(is_np=False)
                print(test_cnt)
                #print(success_rate)

    success_rates = np.mean(success_rates,0)
    print("Mean", np.mean(success_rates,0))
    print("Std", np.std(success_rates,0))

    # each_success_avg = []
    # for i in range(num_oppo):
    #     each_success_avg.append(np.mean(success_rate[i][:n_test]))
    #     print(f'Opponent {i} Success Rate {each_success_avg[i]}')
    # print(f'Total Success Rate {np.mean(each_success_avg)}')

    '''
    global_return_vae_list = []
    global_return_ne_list = []
    global_return_exp3_list = []
    global_return_vae_det_list = []
    global_return_exp3_det_list = []

    policy_vec_list = []
    opponent_list = []

    for n in range(num_oppo):
        obs_list = []
        act_index_list = []
        return_list = []

        policy_vec = policy_vectors[n][0]
        player = this_player

        region = get_p1_region(policy_vec[3:])
        policy_index = region2index(region)
        opponent_policy = get_policy_by_vector(policy_vec,is_best_response=False)
        # NE policy
        ne_response = PolicyKuhn(0,1/3,0,1/3,1/3)

        return_vae_list = []
        return_ne_list = []
        return_best_list = []
        return_exp3_list = []
        return_vae_det_list = []
        return_exp3_det_list = []

        policy_vec_list.append(policy_vec)
        opponent_list.append(region)

        # reset everything every reset_freq=20 opponent - same as change to a new sequence
        # if n%reset_freq == 0:
        exp3.init_weight()
        vi.init_all()

        self_steps = 0
        # all_states = []

        for j in range(n_episode):

            latent = vi.generate_cur_embedding(is_np=False)
            state = game.new_initial_state()
            agent_selected = exp3.sample_action()

            while not state.is_terminal():
                legal_actions = state.legal_actions()
                cur_player = state.current_player()
                if state.is_chance_node():
                    outcomes_with_probs = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes_with_probs)
                    action = np.random.choice(action_list, p=prob_list)
                    state.apply_action(action)
                else:
                    s = state.information_state_tensor(cur_player)
                    if cur_player == player:
                        if agent_selected == 0:
                            # if s not in all_states:
                            #     all_states.append(s)
                            act_vae, act_index_vae, act_prob_vae = agent_vae.select_action(s, latent)
                            action = act_index_vae
                        else:
                            action = ne_response.action(s)
                        self_steps += 1
                    else:
                        action = opponent_policy.action(s)
                        obs_list.append(s)
                        act_index_list.append(action)
                    state.apply_action(action)
            returns = state.returns()
            this_returns = returns[player]
            return_list.append(this_returns)

            exp3.update(this_returns,agent_selected)

            # vi update using the online data (paper version)
            # a replay buffer can also be used to boost the performnace
            if len(obs_list) >= n_steps:
                act_index = np.array(act_index_list).astype(np.float32)
                obs_adv_tensor = torch.FloatTensor(np.array(obs_list[:n_steps]))
                act_adv_tensor = torch.FloatTensor(np.array(act_index[:n_steps]))
                vi.update(obs_adv_tensor, act_adv_tensor)
                emb = vi.generate_cur_embedding(is_np=True)
                ce = vi.get_cur_ce()
                obs_list = []
                act_index_list = []
                return_list = []  

            if (j%(evaluation_freq) == 0 and j > 0) or self_steps >= n_episode:
                emb,sig = vi.get_mu_and_sigma()
                emb_tensor = vi.generate_cur_embedding(is_np=False)
                ce = vi.get_cur_ce()
                p = exp3.get_p()

                avg_return_vae,return_vae = evaluate_vae(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor)
                avg_return_vae_det, return_vae_det = evaluate_vae_det(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor)
                avg_return_ne,return_ne = evaluate_baseline(n_test, player, ne_response, opponent_policy, 'rule_based')
                avg_return_exp3,return_exp3 = evaluate_exp3(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor,ne_response,exp3)
                avg_return_exp3_det,return_exp3_det = evaluate_exp3_det(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor,ne_response,exp3)

                return_vae_list.append(avg_return_vae)
                return_vae_det_list.append(avg_return_vae_det)
                return_ne_list.append(avg_return_ne)
                return_exp3_list.append(avg_return_exp3)
                return_exp3_det_list.append(avg_return_exp3_det)
                global_return_vae_list.append(avg_return_vae)
                global_return_vae_det_list.append(avg_return_vae_det)
                global_return_ne_list.append(avg_return_ne)
                global_return_exp3_list.append(avg_return_exp3)
                global_return_exp3_det_list.append(avg_return_exp3_det)

            if self_steps >= n_episode:
                # print(f'Breaking, {self_steps} steps ({j + 1} episodes) used')
                break

        seq_idx = n//reset_freq
        opp_idx = n%reset_freq
        # print('All states:')
        # for state in sorted(all_states):
        #     print(state)
        # quit()
        # logging.info("seq idx: {}, opp idx: {}, opp name: o{}, gscu: {:.2f}, | gscu det: {:.2f}, | greedy: {:.2f}, | greedy det: {:.2f}, | ne: {:.2f}".format(
        #             seq_idx,opp_idx,region,np.mean(return_exp3_list),np.mean(return_exp3_det_list),np.mean(return_vae_list),np.mean(return_vae_det_list),np.mean(return_ne_list)))
        # print('using', len(return_exp3_list), 'tests')

        if n%reset_freq == (reset_freq-1):
            for l in [global_return_exp3_list, global_return_ne_list, global_return_vae_list]:
                assert len(l) == (n + 1) // reset_freq + 6
                s = 0.0
                for _ in range(7):
                    s += l.pop()
                l.append(s / 7)
            print ('# seq: ', seq_idx, ', total # of opp: ', n + 1,
                   ', avg gscu', np.mean(global_return_exp3_list), np.std(global_return_exp3_list),
                   '| avg gscu det', np.mean(global_return_exp3_det_list), np.std(global_return_exp3_det_list),
                   '| avg greedy', np.mean(global_return_vae_list), np.std(global_return_vae_list),
                   '| avg greedy det', np.mean(global_return_vae_det_list), np.std(global_return_vae_det_list),
                   '| avg ne', np.mean(global_return_ne_list), np.std(global_return_ne_list))
            # print ('-'*10)

            result = {
                'opponent_type':opponent_type,
                'gscu': global_return_exp3_list,
                'greedy': global_return_vae_list,
                'ne': global_return_ne_list,
                'n_opponent':n+1,
                'policy_vec_list':policy_vec_list,
                'opponent_list':opponent_list}

            pickle.dump(result, open(rst_dir+'online_adaption_'+version+'_'+opponent_type+'.p', "wb"))

    print ('version',version)
    print ('opponent_type',opponent_type)
    print ('seed',seed)
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-o', '--opp_type', default='seen', 
                        choices=["seen", "unseen", "mix"], help='type of the opponents')
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-d', '--decoder_file', default='decoder_param_demo.pt', help='vae decoder file')
    parser.add_argument('-r', '--rl_file', default='params_demo.pt', help='conditional RL file')
    parser.add_argument('--player-id', type=int)
    parser.add_argument('--env_config', type=str)
    parser.add_argument('--multi-agent', type=int, default=1)
    parser.add_argument('--all-has-rew-done', action='store_true')
    parser.add_argument('--all-has-last-action', action='store_true')
    parser.add_argument('--num-processes', type=int)
    parser.add_argument('--recurrent-policy', action='store_true')
    parser.add_argument('--GSCU_config', type=str)
    args = parser.parse_args()
    main(args)
