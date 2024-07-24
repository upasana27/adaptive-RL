import os
import random
import time 
import pickle 
import argparse
import numpy as np
from baselines.GSCU.overcooked.utils.config_overcooked import Config
import torch.multiprocessing as mp
from learning.envs import make_vec_envs, make_env
from tqdm import trange


N_TRAINING_SAMPLES = 100000
N_TESTING_SAMPLES = N_TRAINING_SAMPLES//10
SEED_TRAINING = 0
SEED_TESTING = 1
data_dir = Config.DATA_DIR


def rollout_worker(i, num_all_policies, player_id, self_policy, opponent_policy, num_steps, result_queue,
                   args, env_id, seed, rank, log_dir, allow_early_resets):
    env = make_env(args, env_id, seed, rank, log_dir, allow_early_resets)()
    env.set_opponent(opponent_policy)
    env.set_id(player_id)
    self_policy.reset()
    obs = env.reset()
    onehot_label = np.eye(num_all_policies)[i % num_all_policies]
    data_s = []
    data_a = []
    data_i = []
    data_p = []
    for _ in range(num_steps):
        action = self_policy(obs)
        # print(action)
        obs, _, done, info = env.step(action.item())
        data_s.append(info['opponent_obs'])
        data_a.append(info['opponent_act'])
        data_i.append(onehot_label)
        data_p.append(1 - player_id)
        if done:
            obs = env.reset()
            self_policy.reset()
    result_queue.put((data_s, data_a, data_i, data_p))
    env.close()


def main(args):

    profiling = False
    self_inf_time = self_inf_ed = self_inf_st = env_time = None

    mp.set_start_method('spawn')
    testing_mode = [True, False]
    device = 'cpu'
    sample_p1 = Config.SAMPLE_P1_SEEN
    num_policies = len(sample_p1)
    all_onehots = np.eye(num_policies)
    policy_a = Config.CONSERVATIVE_POLICY
    policy_a.device = device
    if args.num_processes is None:
        args.num_processes = num_policies
    elif args.num_processes % num_policies != 0:
        args.num_processes = ((args.num_processes - 1) // num_policies + 1) * num_policies
    print('Using', args.num_processes, 'processes')

    # envs = make_vec_envs(args, 'Overcooked', args.seed, args.num_processes, args.log_dir, device, allow_early_resets=True, use_dummy=True)
    # for i in range(args.num_processes):
    #     envs.env_method('set_opponent', sample_p1[i % num_policies], indices=i)
    # envs.env_method('set_id', args.player_id)

    for is_test in testing_mode:
        if is_test:
            this_version = args.version + '_test'
            n_sample = N_TESTING_SAMPLES
            # seed = SEED_TESTING
            seed = args.seed + 1
            print ('Generating testing data...')
        else:
            this_version = args.version
            n_sample = N_TRAINING_SAMPLES
            seed = args.seed
            # seed = SEED_TRAINING
            print ('Generating training data...')

        np.random.seed(seed)
        random.seed(seed)

        # policy_a.reset()

        data_s = []
        data_a = []
        data_i = []
        data_p = []

        start_time = time.time()
        n_steps = n_sample // args.num_processes
        result_queues = [mp.Queue() for _ in range(args.num_processes)]
        procs = [mp.Process(target=rollout_worker, args=(i, num_policies, args.player_id, policy_a,
                                                         sample_p1[i % num_policies], n_steps, result_queues[i],
                                                         args, 'Overcooked', args.seed, i, None, False))
                 for i in range(args.num_processes)]
        for p in procs:
            p.start()
        # print('Collecting', n_sample, 'steps')
        #
        # if profiling:
        #     self_inf_time = 0.0
        #     env_time = 0.0
        # obs = envs.reset()
        # for steps in trange(n_steps):
        #
        #     if profiling:
        #         self_inf_st = time.time()
        #
        #     action = policy_a(obs)
        #
        #     if profiling:
        #         self_inf_ed = time.time()
        #
        #     obs, _, _, infos = envs.step(action)
        #
        #     if profiling:
        #         env_ed = time.time()
        #         self_inf_time += self_inf_ed - self_inf_st
        #         env_time += env_ed - self_inf_ed
        #
        #     for i in range(len(sample_p1)):
        #         data_s.append(infos[i]['opponent_obs'])
        #         data_a.append(infos[i]['opponent_act'])
        #         data_i.append(all_onehots[i])
        #         data_p.append(1 - args.player_id)
        #
        # if profiling:
        #     envs.env_method('print_time')
        #     print('In main, self inf time:', self_inf_time, 'env time:', env_time)
        #     quit()
        for rq in result_queues:
            data_s_, data_a_, data_i_, data_p_ = rq.get()
            data_s += data_s_
            data_a += data_a_
            data_i += data_i_
            data_p += data_p_

        for p in procs:
            p.join()

        end_time = time.time()
        print('Takes {:.2f}s to generate {} steps, fps {:.0f}'.format(end_time-start_time, n_sample, n_sample / (end_time - start_time)))

        vae_data = {
            'data_s': data_s,
            'data_a': data_a,
            'data_i': data_i,
            'data_p': data_p
        }

        os.makedirs(data_dir, exist_ok=True)
        pickle.dump(vae_data, open(os.path.join(data_dir, 'vae_data_overcooked_'+this_version+'.p'), "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0')
    parser.add_argument('-seed', '--seed', type=int)
    parser.add_argument('--player-id', type=int)
    parser.add_argument('--env_config', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--multi-agent', type=int, default=1)
    parser.add_argument('--num-processes', type=int)
    parser.add_argument('--all-has-rew-done', action='store_true')
    parser.add_argument('--recurrent-policy', action='store_true')
    arg = parser.parse_args()

    main(arg)

