import pickle
import numpy as np


if __name__ == '__main__':
    all_run_data = []
    for seed in range(3):
        file_name = f'./results/online_test/online_adaption_pp_1m_v{seed}_unseen.p'
        with open(file_name, 'rb') as f:
            data = pickle.load(f)['gscu']
        data = np.array(data).reshape(10, 24, 5)
        data = data.mean(axis=1)
        data = data.mean(axis=0)
        all_run_data.append(data)
        # print(np.mean(data, axis=0), np.std(data, axis=0))
    all_run_data = np.array(all_run_data)
    print(all_run_data.shape)
    print(np.mean(all_run_data, axis=0), np.std(all_run_data, axis=0))
    all_run_data = all_run_data.mean(axis=1)
    print(np.mean(all_run_data), np.std(all_run_data))
