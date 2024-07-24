import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
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
from online_test.bayesian_update import VariationalInference,EXP3
from conditional_RL.conditional_rl_model import PPO_VAE
from utils.config_kuhn_poker import Config
from utils.mypolicy import PolicyKuhn
from environment.kuhn_poker.kuhn_poker_oppo_hand import KuhnPoker_SingleEnv
from environment.kuhn_poker.policy_new import KuhnPokerPolicy
import pickle
import re

np.set_printoptions(precision=6)
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def print(*args):
    __builtins__.print(*("%.4f" % a if isinstance(a, float) else a
                         for a in args))
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
        obs = [0.0] * 13
        obs[i // 3] = 1.0
        obs[7 + (i % 3)] = 1.0

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
        obs = [0.0] * 13
        obs[i // 3] = 1.0
        obs[7 + (i % 3)] = 1.0

        _, action, action_prob = agent_vae.select_action(obs.copy(), latent)
        if action == 0:
            action_prob = 1.0 - action_prob
        assert 0.0 <= action_prob <= 1.0
        vae_probs.append(action_prob)
    vae_return = opponent_policy.get_return_complex(vae_probs)
    ne_return = opponent_policy.get_return(ne_response.alpha, ne_response.beta, ne_response.gamma)
    exp3.generate_p()
    return vae_return * exp3.p[0] + ne_return * (1.0 - exp3.p[0]), None
    # game = pyspiel.load_game("kuhn_poker(players=2)")
    # return_list = []
    # for _ in range(n_test):
    #     state = game.new_initial_state()
    #     agent_selected = exp3.sample_action()
    #     while not state.is_terminal():
    #         legal_actions = state.legal_actions()
    #         cur_player = state.current_player()
    #         if state.is_chance_node():
    #             outcomes_with_probs = state.chance_outcomes()
    #             action_list, prob_list = zip(*outcomes_with_probs)
    #             action = np.random.choice(action_list, p=prob_list)
    #             state.apply_action(action)
    #         else:
    #             s = state.information_state_tensor(cur_player)
    #             if cur_player == player:
    #                 if agent_selected == 0:
    #                     _, action, _ = agent_vae.select_action(s, latent)
    #                 else:
    #                     action = ne_response.action(s)
    #             else:
    #                 if opponent_type == 'rl':
    #                     _, action, _ = opponent_policy.select_action(s)
    #                 else:
    #                     action = opponent_policy.action(s)
    #             state.apply_action(action)
    #     returns = state.returns()
    #     this_returns = returns[player]
    #     return_list.append(this_returns)
    # return np.mean(return_list),return_list

def evaluate_vae_det(n_test, player, agent_vae, opponent_policy, opponent_type, latent):
    vae_probs = []
    for i in range(6):
        obs = [0.0] * 13
        obs[i // 3] = 1.0
        obs[7 + (i % 3)] = 1.0

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
        obs = [0.0] * 13
        obs[i // 3] = 1.0
        obs[7 + (i % 3)] = 1.0

        _, action, action_prob = agent_vae.select_action(obs.copy(), latent)
        if action == 0:
            action_prob = 1.0 - action_prob
        assert 0.0 <= action_prob <= 1.0
        vae_probs.append(action_prob)
    vae_return = opponent_policy.get_return_complex(vae_probs)
    return vae_return, None
    # game = pyspiel.load_game("kuhn_poker(players=2)")
    # return_list = []
    # for _ in range(n_test):
    #     state = game.new_initial_state()
    #     while not state.is_terminal():
    #         legal_actions = state.legal_actions()
    #         cur_player = state.current_player()
    #         if state.is_chance_node():
    #             outcomes_with_probs = state.chance_outcomes()
    #             action_list, prob_list = zip(*outcomes_with_probs)
    #             action = np.random.choice(action_list, p=prob_list)
    #             state.apply_action(action)
    #         else:
    #             s = state.information_state_tensor(cur_player)
    #             if cur_player == player:
    #                 _, action, _ = agent_vae.select_action(s, latent)
    #             else:
    #                 if opponent_type == 'rl':
    #                     _, action, _ = opponent_policy.select_action(s)
    #                 else:
    #                     action = opponent_policy.action(s)
    #             state.apply_action(action)
    #     returns = state.returns()
    #     this_returns = returns[player]
    #     return_list.append(this_returns)
    # return np.mean(return_list),return_list

def evaluate_baseline(n_test, player, response, opponent_policy, opponent_type):
    ne_return = opponent_policy.get_return(response.alpha, response.beta, response.gamma)
    return ne_return, None
    # game = pyspiel.load_game("kuhn_poker(players=2)")
    # return_list = []
    # for _ in range(n_test):
    #     state = game.new_initial_state()
    #     while not state.is_terminal():
    #         legal_actions = state.legal_actions()
    #         cur_player = state.current_player()
    #         if state.is_chance_node():
    #             outcomes_with_probs = state.chance_outcomes()
    #             action_list, prob_list = zip(*outcomes_with_probs)
    #             action = np.random.choice(action_list, p=prob_list)
    #             state.apply_action(action)
    #         else:
    #             s = state.information_state_tensor(cur_player)
    #             if cur_player == player:
    #                 action = response.action(s)
    #             else:
    #                 if opponent_type == 'rl':
    #                     _, action, _ = opponent_policy.select_action(s)
    #                 else:
    #                     action = opponent_policy.action(s)
    #             state.apply_action(action)
    #     returns = state.returns()
    #     this_returns = returns[player]
    #     return_list.append(this_returns)
    # return np.mean(return_list),return_list


def main(args):

    state_dim = Config.OBS_DIM
    action_dim = Config.ACTION_DIM
    n_adv_pool = Config.NUM_ADV_POOL
    embedding_dim = Config.LATENT_DIM
    hidden_dim = Config.HIDDEN_DIM

    n_steps = 10 # vi update freq
    this_player = 0 # controlling player
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
    n_pass = 200
    # n_test = None  # Replacing trials with expected reward computation, so this is not needed
    n_episode = 100
    # evaluation_freq = 50

    version = args.version 
    opponent_type = args.opp_type
    print ('opponent_type',opponent_type)

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    encoder_weight_path = Config.VAE_MODEL_DIR + args.encoder_file
    decoder_weight_path = Config.VAE_MODEL_DIR + args.decoder_file
    conditional_rl_weight_path = Config.RL_MODEL_DIR + args.rl_file
    opponent_model = OpponentModel(state_dim, n_adv_pool, hidden_dim, embedding_dim, action_dim, encoder_weight_path, decoder_weight_path)
    vi = VariationalInference(opponent_model, latent_dim=embedding_dim, n_update_times=50, game_steps=n_steps)
    exp3 = EXP3(n_action=2, gamma=0.3, min_reward=-2, max_reward=2) # lr of exp3 is set to 0.3
    agent_vae = PPO_VAE(state_dim, hidden_dim, embedding_dim, action_dim, 0.0, 0.0, encoder_weight_path, n_adv_pool)
    agent_vae.init_from_save(conditional_rl_weight_path)

    rst_dir = Config.ONLINE_TEST_RST_DIR
    data_dir = Config.DATA_DIR
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=False) 
            
    # a randomly generated sequence is used. You can create your own.
    # policy_vectors_df = pd.read_pickle(data_dir+'online_test_policy_vectors_demo.p')
    # if opponent_type == 'seen':
    #     policy_vectors = policy_vectors_df['seen']
    # elif opponent_type == 'unseen':
    #     policy_vectors = policy_vectors_df['unseen']
    # elif opponent_type == 'mix':
    #     policy_vectors = policy_vectors_df['mix']
    # else:
    #     raise ValueError('No such opponent type')

    if opponent_type == 'seen':
        opponent_policies = Config.SAMPLE_P1_SEEN
        # policy_vectors = [[[0., 1. / 3., 0., p[0], p[1]], None] for p in Config.SAMPLE_P1_SEEN(seed)]
    elif opponent_type == 'eval':
        opponent_policies = Config.SAMPLE_P1_EVAL
    elif opponent_type == 'unseen':
        opponent_policies = Config.SAMPLE_P1_UNSEEN
        # policy_vectors = [[[0., 1. / 3., 0., p[0], p[1]], None] for p in Config.SAMPLE_P1_UNSEEN]
    else:
        raise NotImplementedError
    n_opponent = len(opponent_policies)
    print('Testing against', n_opponent, 'opponents for', n_pass, 'pass')

    env = KuhnPoker_SingleEnv()
    # final_result = []
    # final_cumul_result = []

    actual_returns = []

    for pass_idx in range(n_pass):

        # global_return_vae_list = []
        # global_return_ne_list = []
        # global_return_exp3_list = []
        # global_return_vae_det_list = []
        # global_return_exp3_det_list = []
        # global_return_run_list = []
        # global_return_cumul_run_list = []

        policy_vec_list = []
        opponent_list = []
        actual_returns.append([])

        for n in range(n_opponent):
            actual_returns[-1].append([])
            obs_list = []
            act_index_list = []
            # return_list = []

            player = this_player
            opponent_policy = opponent_policies[n]
            env.set_opponent(opponent_policy)
            # NE policy
            ne_response = KuhnPokerPolicy([1/3,1/3])
            ne_response.alpha = 0.0
            ne_response.beta = 1. / 3.
            ne_response.gamma = 0.0

            # return_vae_list = []
            # return_ne_list = []
            # return_best_list = []
            # return_exp3_list = []
            # return_vae_det_list = []
            # return_exp3_det_list = []
            # return_run_list = []

            # reset everything every reset_freq=20 opponent - same as change to a new sequence
            # if n%reset_freq == 0:
            exp3.init_weight()
            vi.init_all()

            # self_steps = 0
            # all_states = []

            for j in range(n_episode):

                latent = vi.generate_cur_embedding(is_np=False)
                obs = env.reset()
                agent_selected = exp3.sample_action()
                agent_selected = 0
                episode_return = 0.0

                while True:
                    if agent_selected == 0:
                        # if s not in all_states:
                        #     all_states.append(s)
                        act_vae, act_index_vae, act_prob_vae = agent_vae.select_action(obs, latent)
                        action = act_index_vae
                    else:
                        action = ne_response(obs)

                    obs, rew, done, info = env.step(action)
                    episode_return += rew
                    if 'opponent_obs' in info:
                        obs_list.append(info['opponent_obs'])
                        act_index_list.append(info['opponent_act'])
                    if done:
                        break

                this_returns = episode_return
                actual_returns[-1][-1].append(this_returns)
                # return_list.append(this_returns)
                # return_run_list.append(this_returns)
                # global_return_run_list.append(this_returns)

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
                    # return_list = []

                # if j % evaluation_freq == evaluation_freq - 1:
                #     emb,sig = vi.get_mu_and_sigma()
                #     emb_tensor = vi.generate_cur_embedding(is_np=False)
                #     ce = vi.get_cur_ce()
                #     p = exp3.get_p()
                #
                #     avg_return_vae,return_vae = evaluate_vae(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor)
                #     avg_return_vae_det, return_vae_det = evaluate_vae_det(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor)
                #     avg_return_ne,return_ne = evaluate_baseline(n_test, player, ne_response, opponent_policy, 'rule_based')
                #     avg_return_exp3,return_exp3 = evaluate_exp3(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor,ne_response,exp3)
                #     avg_return_exp3_det,return_exp3_det = evaluate_exp3_det(n_test, player, agent_vae, opponent_policy, 'rule_based', emb_tensor,ne_response,exp3)
                #
                #     return_vae_list.append(avg_return_vae)
                #     return_vae_det_list.append(avg_return_vae_det)
                #     return_ne_list.append(avg_return_ne)
                #     return_exp3_list.append(avg_return_exp3)
                #     return_exp3_det_list.append(avg_return_exp3_det)
                #     global_return_vae_list.append(avg_return_vae)
                #     global_return_vae_det_list.append(avg_return_vae_det)
                #     global_return_ne_list.append(avg_return_ne)
                #     global_return_exp3_list.append(avg_return_exp3)
                #     global_return_exp3_det_list.append(avg_return_exp3_det)

                # if self_steps >= n_episode:
                #     # print(f'Breaking, {self_steps} steps ({j + 1} episodes) used')
                #     break

            print('Test pass', pass_idx, 'opponent', n, ', actual', np.mean(actual_returns[-1][-1]), np.std(np.mean(actual_returns[-1][-1])))
            # print ('Test pass', pass_idx, 'opponent', n,
            #        ', avg gscu', np.mean(return_exp3_list), np.std(return_exp3_list),
            #        '| avg gscu det', np.mean(return_exp3_det_list), np.std(return_exp3_det_list),
            #        '| avg greedy', np.mean(return_vae_list), np.std(return_vae_list),
            #        '| avg greedy det', np.mean(return_vae_det_list), np.std(return_vae_det_list),
            #        '| avg ne', np.mean(return_ne_list), np.std(return_ne_list),
            #        '| actual', np.mean(return_run_list), np.std(return_run_list))
            # print(len(return_exp3_list))
            # print(len(return_run_list))
            # global_return_cumul_run_list.append(return_run_list.copy())

            # seq_idx = n//reset_freq
            # opp_idx = n%reset_freq
            # print('All states:')
            # for state in sorted(all_states):
            #     print(state)
            # quit()
            # logging.info("seq idx: {}, opp idx: {}, opp name: o{}, gscu: {:.2f}, | gscu det: {:.2f}, | greedy: {:.2f}, | greedy det: {:.2f}, | ne: {:.2f}".format(
            #             seq_idx,opp_idx,region,np.mean(return_exp3_list),np.mean(return_exp3_det_list),np.mean(return_vae_list),np.mean(return_vae_det_list),np.mean(return_ne_list)))
            # print('using', len(return_exp3_list), 'tests')

            # if n%reset_freq == (reset_freq-1):
            #     for l in [global_return_exp3_list, global_return_ne_list, global_return_vae_list]:
            #         assert len(l) == (n + 1) // reset_freq + 6
            #         s = 0.0
            #         for _ in range(7):
            #             s += l.pop()
            #         l.append(s / 7)
            #     print ('# seq: ', seq_idx, ', total # of opp: ', n + 1,
            #            ', avg gscu', np.mean(global_return_exp3_list), np.std(global_return_exp3_list),
            #            '| avg gscu det', np.mean(global_return_exp3_det_list), np.std(global_return_exp3_det_list),
            #            '| avg greedy', np.mean(global_return_vae_list), np.std(global_return_vae_list),
            #            '| avg greedy det', np.mean(global_return_vae_det_list), np.std(global_return_vae_det_list),
            #            '| avg ne', np.mean(global_return_ne_list), np.std(global_return_ne_list))
            #     # print ('-'*10)
            #
            #     result = {
            #         'opponent_type':opponent_type,
            #         'gscu': global_return_exp3_list,
            #         'greedy': global_return_vae_list,
            #         'ne': global_return_ne_list,
            #         'n_opponent':n+1,
            #         'policy_vec_list':policy_vec_list,
            #         'opponent_list':opponent_list}
            #
            #     pickle.dump(result, open(rst_dir+'online_adaption_'+version+'_'+opponent_type+'.p', "wb"))

        print ('version',version)
        print ('opponent_type',opponent_type)
        print ('seed',seed)
        print('Pass', pass_idx, 'result', ', avg actual', np.mean(actual_returns[-1]), np.std(actual_returns[-1]))
        # print('Pass', pass_idx, 'result',
        #        ', avg gscu', np.mean(global_return_exp3_list), np.std(global_return_exp3_list),
        #        '| avg gscu det', np.mean(global_return_exp3_det_list), np.std(global_return_exp3_det_list),
        #        '| avg greedy', np.mean(global_return_vae_list), np.std(global_return_vae_list),
        #        '| avg greedy det', np.mean(global_return_vae_det_list), np.std(global_return_vae_det_list),
        #        '| avg ne', np.mean(global_return_ne_list), np.std(global_return_ne_list),
        #        '| avg actual', np.mean(global_return_run_list), np.std(global_return_run_list))
        # print(len(global_return_ne_list))
        # print(len(global_return_run_list))
        # final_result.append(np.mean(global_return_run_list))
        # final_cumul_result.append(np.mean(global_return_cumul_run_list, axis=0))
        # print('Current running final result:', np.mean(final_result), np.std(final_result), len(final_result))

    actual_returns = np.array(actual_returns)
    print('Final result of', n_pass, 'passes:', np.mean(actual_returns), np.std(actual_returns))
    match = re.fullmatch(r'encoder_vae_param_same_env_40_sd(\d)_19.pt', args.encoder_file)
    assert match is not None, f'Wrong encoder file name: {args.encoder_file}'
    train_seed = int(match.group(1))
    with open(f'./ppo_gscu_greedy_seed{train_seed}_{n_pass}pass_all_results.pkl', 'wb') as f:
        pickle.dump(actual_returns, f)
    # print(len(final_result))
    # if not os.path.exists(Config.ONLINE_TEST_RST_DIR):
    #     os.makedirs(Config.ONLINE_TEST_RST_DIR)
    # np.save(os.path.join(Config.ONLINE_TEST_RST_DIR, 'all_results.npy'), final_cumul_result)
    print('Results saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0', help='version')
    parser.add_argument('-seed', '--seed', default=0, help='seed')
    parser.add_argument('-o', '--opp_type', default='seen', 
                        choices=["seen", "unseen", "mix", 'eval'], help='type of the opponents')
    parser.add_argument('-e', '--encoder_file', default='encoder_vae_param_demo.pt', help='vae encoder file')
    parser.add_argument('-d', '--decoder_file', default='decoder_param_demo.pt', help='vae decoder file')
    parser.add_argument('-r', '--rl_file', default='params_demo.pt', help='conditional RL file')
    args = parser.parse_args()
    main(args)
