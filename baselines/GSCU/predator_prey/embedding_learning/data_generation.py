#!/usr/bin/env python
import pickle
import time
import torch
from environment.mpe.policy_both import get_train_eval_pool
from learning.envs import make_env
from baselines.GSCU.predator_prey.utils.config_predator_prey import Config
import numpy as np
import random
import argparse
from tqdm import trange


# N_ADV = 3

N_TRAINING_EPISODES = 20000
N_TESTING_EPISODES = N_TRAINING_EPISODES//10
SEED_TRAINING = 0
SEED_TESTING = 1

# ramdomly generate 3 adv policies and a agent 
# def generate_policies(env, adv_pool,agent_pool):
#     selected_advs_ids = np.random.choice(range(0, len(adv_pool)), size=N_ADV, replace=True)
#     selected_agent_ids = np.random.choice(range(0, len(agent_pool)), size=1, replace=True)
#     adv_policies = []
#     agent_policies = []
#     for idx, adv_id in enumerate(selected_advs_ids):
#         policy = get_policy_by_name(env,adv_pool[adv_id],idx)
#         adv_policies.append(policy)
#     agent_policies = [eval(agent_pool[selected_agent_ids[0]] + "(env," + str(N_ADV) +")")]
#     policies = adv_policies + agent_policies
#     return policies, selected_advs_ids
#
# def get_all_adv_policies(env,adv_pool,agent_index):
#     all_policies = []
#     for adv_id in range(len(adv_pool)):
#         policy = get_policy_by_name(env,adv_pool[adv_id],agent_index)
#         all_policies.append(policy)
#     return all_policies
#
# def get_policy_by_name(env,policy_name,agent_index):
#     return eval(policy_name + "(env," + str(agent_index) +")")

def main(version):
    # # load scenario from script
    # scenario = scenarios.load(args.scenario).Scenario()
    # # create world
    # world = scenario.make_world()
    # # create multiagent environment
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
    #                     scenario.observation, info_callback=None,
    #                     shared_viewer=False, discrete_action=True)
    from baselines.GSCU.predator_prey.utils.config_predator_prey import args
    args.version = version
    env = make_env(args, 'MPE', 0, 0, None, False)()
    generalist_path = '../../../../data/PredatorPrey/final_checkpoints/ppo_test_generalist_seed1.pt'
    ego_agent = torch.load(generalist_path, map_location=args.device).actors[0]
    print(ego_agent)
    # quit()

    data_dir = Config.DATA_DIR

    train_pool, _ = get_train_eval_pool(args)
    print(train_pool)
    # quit()

    version = args.version
    
    testing_mode = [True, False]
    for is_test in testing_mode:
        if is_test:
            this_version = version + '_test'
            eposides = N_TESTING_EPISODES
            seed = SEED_TESTING
            print ('Generating testing data...')
        else:
            this_version = version
            eposides = N_TRAINING_EPISODES
            seed = SEED_TRAINING
            print ('Generating training data...')

        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

        data_s = []
        data_a = []
        data_i = []
        n_step = 0
        n_sample = 0

        n_same_output = 0
        n_all_output = 0

        for e in trange(eposides):
            train_peer_idx = e % len(train_pool)
            env.set_opponent(train_pool[train_peer_idx])
            env.set_id(args.player_id)
            # policies, adv_ids = generate_policies(env, adv_pool,agent_pool)

            # execution loop
            env.full_reset()
            obs = env.reset()
            rnn_states = None if ego_agent.rnn is None else torch.zeros(1, ego_agent.rnn.out_dim, device=args.device)
            # print(obs.shape, rnn_states.shape)
            # obs_n = env._reset()
            for st in range(args.horizon):
                start = time.time()
                # query for action from each agent's policy
                with torch.no_grad():
                    action, _, rnn_states, _ = ego_agent.act(torch.from_numpy(obs).float().unsqueeze(0).to(args.device),
                                                             rnn_states,
                                                             torch.ones(1, 1, device=args.device),
                                                             None, deterministic=False)
                # print(action.shape, rnn_states.shape)
                # act_n = []
                # for i, policy in enumerate(policies):
                #     act = policy.action(obs_n[i])
                #     act_n.append(act)
                #     this_action = []
                #     # collect the obs/a/policy_index for opponent policies
                #     if i < N_ADV:
                #         # simulate the action for all opponent in the pool
                #         for adv_id, sudo_policy in enumerate(all_policies[i]):
                #             sudo_act = sudo_policy.action(obs_n[i])
                #             data_s.append(obs_n[i])
                #             data_a.append(sudo_act)
                #             data_i.append(adv_id)
                #
                #             this_action.append(sudo_act)
                #             n_sample += 1
                #         n_all_output += 1

                # step environment
                # obs_n, reward_n, done_n, _ = env._step(act_n)
                obs, _, done, info = env.step(action.cpu().numpy())
                for a, o in zip(info['opponent_act'], np.split(info['opponent_obs'], args.num_agents - 1)):
                    data_s.append(o)
                    data_a.append(a)
                    data_i.append(train_peer_idx)
                    n_sample += 1
                n_all_output += 1
                # print(info, len(info['opponent_act']), info['opponent_obs'].shape)
                # quit()
                n_step += 1

            if (e+1) % (eposides//10) == 0:
                print ('n eposides',e+1, ' | n sample',n_sample)


        vae_data = {
            'data_s': data_s,
            'data_a': data_a,
            'data_i': data_i}

        pickle.dump(vae_data, 
            open(data_dir+'vae_data_' + this_version + '.p', "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-s', '--scenario', default='simple_tag_partial.py', help='Path of the scenario Python script')
    # parser.add_argument('-st', '--steps', default=50, type=int, help='Num of steps in a single run')
    parser.add_argument('-v', '--version', required=True)
    args = parser.parse_args()
    main(args.version)
