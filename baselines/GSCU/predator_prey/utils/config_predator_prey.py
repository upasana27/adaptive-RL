# -*- coding:utf-8 -*-
from argparse import Namespace

class Config:

    HIDDEN_DIM = 128
    LATENT_DIM = 2
    WINDOW_SIZW = 8

    ADV_POOL_SEEN = ['PolicyN', 'PolicyNW', 'PolicyW', 'PolicySW']
    ADV_POOL_UNSEEN = ['PolicyNE','PolicySE', 'PolicyE', 'PolicyS']
    ADV_POOL_MIX = ADV_POOL_SEEN + ADV_POOL_UNSEEN

    DATA_DIR = '../data/'
    VAE_MODEL_DIR = '../model_params/VAE/'
    VAE_RST_DIR = '../results/VAE/'

    RL_MODEL_DIR = '../model_params/RL/'
    RL_TRAINING_RST_DIR = '../results/RL/'

    ONLINE_TEST_RST_DIR = '../results/online_test/'

    OPPONENT_MODEL_DIR = '../model_params/opponent/'


args = Namespace(
    scenario='simple_tag_multi_partial',
    num_agents=4,
    num_good_agents=2,
    obs_radius=0.2,
    init_radius=1.0,
    horizon=50,
    shaped_reward=True,
    collide_reward=False,
    collide_reward_once=True,
    visit_reward_coef=None,
    shuffle_agents=False,
    all_has_all_time_steps=False,
    watch_tower=True,
    pool_seed=1,
    player_id=0,
    separate_patterns=True,
    history_size=5,
    all_has_rew_done=False,
    all_has_last_action=False,
    train_pool_size=16,
    eval_pool_size=24,
    device='cuda:0'
)



