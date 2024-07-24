import os
import random
import time 
import pickle 
import argparse
import numpy as np
import sys
sys.path.append('.')
import torch.multiprocessing as mp
from learning.envs import make_vec_envs, make_env
from tqdm import trange



def main(args):
    if args.GSCU_config == "Hard":
        from baselines.GSCU.overcooked.utils.config_overcooked import Config_Hard as Config
    else:
        from baselines.GSCU.overcooked.utils.config_overcooked import Config
    N_TRAINING_SAMPLES = 2000000
    N_TESTING_SAMPLES = N_TRAINING_SAMPLES//10
    SEED_TRAINING = 0
    SEED_TESTING = 1
    data_dir = Config.DATA_DIR


    profiling = True
    self_inf_time = self_inf_ed = self_inf_st = env_time = None

    testing_mode = [True, False]
    device = 'cuda'
    sample_p1 = Config.SAMPLE_P1_SEEN
    num_policies = len(sample_p1)
    all_onehots = np.eye(num_policies)
    #action_onehots = np.eye(Config.ACTION_DIM)
    policy_a = Config.CONSERVATIVE_POLICY
    policy_a.device = device
    assert args.num_processes is None
    if args.num_processes is None:
        args.num_processes = num_policies
    elif args.num_processes % num_policies != 0:
        args.num_processes = ((args.num_processes - 1) // num_policies + 1) * num_policies
    print('Using', args.num_processes, 'processes')

    envs = make_vec_envs(args, 'Overcooked', args.seed, args.num_processes, args.log_dir, device,
                         allow_early_resets=True, always_use_dummy=False)
    for i in range(args.num_processes):
        envs.env_method('set_opponent', sample_p1[i % num_policies], indices=i)
    envs.env_method('set_id', args.player_id)

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

        policy_a.reset()

        data_s = []
        data_a = []
        data_i = []
        data_p = []

        start_time = time.time()
        n_steps = n_sample // args.num_processes
        print('Collecting', n_sample, 'steps')

        if profiling:
            self_inf_time = 0.0
            env_time = 0.0
        obs = envs.reset()
        for _ in trange(n_steps):

            if profiling:
                self_inf_st = time.time()

            action = policy_a(obs)

            if profiling:
                self_inf_ed = time.time()

            obs, _, _, infos = envs.step(action)

            if profiling:
                env_ed = time.time()
                self_inf_time += self_inf_ed - self_inf_st
                env_time += env_ed - self_inf_ed

            for i in range(len(sample_p1)):
                if device == 'cuda':
                    oppo_obs = infos[i]['opponent_obs'].cpu().numpy()
                    oppo_act = infos[i]['opponent_act'].cpu().numpy()
                data_s.append(oppo_obs)
                data_a.append(oppo_act)
                data_i.append(all_onehots[i])
                data_p.append(1 - args.player_id)

        if profiling:
            envs.env_method('print_time')
            print('In main, self inf time:', self_inf_time, 'env time:', env_time)
            #quit()

        end_time = time.time()
        print('Takes {:.2f}s to generate {} steps, fps {:.0f}'.format(end_time-start_time, n_sample,
                                                                      n_sample / (end_time - start_time)))

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
    parser.add_argument('--all-has-last-action', action='store_true')
    parser.add_argument('--recurrent-policy', action='store_true')
    parser.add_argument('--GSCU_config', type=str)
    arg = parser.parse_args()

    main(arg)

