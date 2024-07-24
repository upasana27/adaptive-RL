import os.path
import random
import time 
import pickle 
import argparse
import numpy as np
from baselines.GSCU.kuhn_poker.utils.config_kuhn_poker import Config
from environment.kuhn_poker.kuhn_poker_oppo_hand import KuhnPoker_SingleEnv
from environment.kuhn_poker.policy_new import KuhnPokerPolicy


def main(version, raw_seed):
    N_TRAINING_SAMPLES = 1000000
    N_TESTING_SAMPLES = N_TRAINING_SAMPLES // 10
    # SEED_TRAINING = 0
    # SEED_TESTING = 1
    data_dir = Config.DATA_DIR

    testing_mode = [True, False]
    sample_p1 = Config.SAMPLE_P1_SEEN
    for is_test in testing_mode:
        if is_test:
            this_version = version + '_test'
            n_sample = N_TESTING_SAMPLES
            # seed = SEED_TESTING
            seed = raw_seed + 1
            print ('Generating testing data...')
        else:
            this_version = version
            n_sample = N_TRAINING_SAMPLES
            seed = raw_seed
            # seed = SEED_TRAINING
            print ('Generating training data...')

        np.random.seed(seed)
        random.seed(seed)

        env = KuhnPoker_SingleEnv()

        data_s = []
        data_a = []
        data_i = []
        data_p = []

        start_time = time.time()
        for i in range(n_sample):

            # p0 is with random policy parameters
            policy_vector_a = np.random.rand(5)
            policy_a = KuhnPokerPolicy(policy_vector_a)

            # p1 is randomly sampled from the seen p1 pool
            n_class = len(sample_p1)
            rand_int = np.random.randint(n_class)
            p1_idx_onehot = np.eye(n_class)[rand_int]
            policy_b = sample_p1[rand_int]
            env.set_opponent(policy_b)

            obs = env.reset()

            while True:
                action = policy_a(obs)
                obs, rew, done, info = env.step(action)
                if 'opponent_obs' in info:
                    data_s.append(info['opponent_obs'])
                    data_a.append(info['opponent_act'])
                    data_i.append(p1_idx_onehot)
                    data_p.append(1)
                if done:
                    break

        end_time = time.time()
        print ('Time dur: {:.2f}s'.format(end_time-start_time))
        vae_data = {
            'data_s': data_s,
            'data_a': data_a,
            'data_i': data_i,
            'data_p': data_p}

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(os.path.join(data_dir, 'vae_data_kuhn_poker_'+this_version+'.p'), "wb") as f:
            pickle.dump(vae_data, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--version', default='v0')
    parser.add_argument('-seed', '--seed', type=int)
    args = parser.parse_args()

    main(args.version, args.seed)

