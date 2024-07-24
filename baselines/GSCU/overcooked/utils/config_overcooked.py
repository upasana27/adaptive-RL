# -*- coding:utf-8 -*-
from environment.overcooked.policy import load_good_self_play_policy_pool, PretrainedPolicy, generate_policy_pool


class Config:

    OBS_DIM = 64
    NUM_ADV_POOL = 25
    ACTION_DIM = 6
    HIDDEN_DIM = 128
    LATENT_DIM = 2
    PLAYER_ID = 0

    #SAMPLE_P1_SEEN, SAMPLE_P1_UNSEEN = load_good_self_play_policy_pool(1 - PLAYER_ID)
    SAMPLE = generate_policy_pool(0, 1, "full_divider_potato", 40)
    SAMPLE_P1_SEEN = SAMPLE[:25]
    TEST_OPPO = SAMPLE[:12]
    CONSERVATIVE_POLICY = PretrainedPolicy(
        './data/Overcooked/br_checkpoints/joint_potato_final.pt', PLAYER_ID,
        is_self_play=False, batch_size=len(SAMPLE_P1_SEEN)
    )
    ONLINE_CONSERVATIVE_POLICY = PretrainedPolicy(
        './data/Overcooked/br_checkpoints/rule_potato_opp1_final.pt', PLAYER_ID,
        is_self_play=False, batch_size=1
    )

    DATA_DIR = './baselines/GSCU/overcooked/data/'
    VAE_MODEL_DIR = './baselines/GSCU/overcooked/model_params/VAE/'
    VAE_RST_DIR = './baselines/GSCU/overcooked/results/VAE/'

    RL_MODEL_DIR = './baselines/GSCU/overcooked/model_params/RL/'
    RL_TRAINING_RST_DIR = './baselines/GSCU/overcooked/results/RL/'

    ONLINE_TEST_RST_DIR = './baselines/GSCU/overcooked/results/online_test/'

    OPPONENT_MODEL_DIR = './baselines/GSCU/overcooked/model_params/opponent/'

class Config_Hard:

    OBS_DIM = 64
    NUM_ADV_POOL = 25
    ACTION_DIM = 6
    HIDDEN_DIM = 128
    LATENT_DIM = 2
    PLAYER_ID = 0

    #SAMPLE_P1_SEEN, SAMPLE_P1_UNSEEN = load_good_self_play_policy_pool(1 - PLAYER_ID)
    SAMPLE = generate_policy_pool(0, 1, "full_divider_potato", 50)
    SAMPLE_P1_SEEN = SAMPLE[:25]
    TEST_OPPO = SAMPLE[32:48]
    CONSERVATIVE_POLICY = PretrainedPolicy(
        './data/Overcooked/br_checkpoints/joint_potato_hard_30m.pt', PLAYER_ID,
        is_self_play=False, batch_size=len(SAMPLE_P1_SEEN)
    )
    ONLINE_CONSERVATIVE_POLICY = PretrainedPolicy(
        './data/Overcooked/br_checkpoints/joint_potato_hard_30m.pt', PLAYER_ID,
        is_self_play=False, batch_size=1
    )

    DATA_DIR = './baselines/GSCU/overcooked/data/hard/'
    VAE_MODEL_DIR = './baselines/GSCU/overcooked/model_params/VAE/hard/'
    VAE_RST_DIR = './baselines/GSCU/overcooked/results/VAE/hard/'

    RL_MODEL_DIR = './baselines/GSCU/overcooked/model_params/RL/hard/'
    RL_TRAINING_RST_DIR = './baselines/GSCU/overcooked/results/RL/hard/'

    ONLINE_TEST_RST_DIR = './baselines/GSCU/overcooked/results/online_test/hard/'

    OPPONENT_MODEL_DIR = './baselines/GSCU/overcooked/model_params/opponent/hard/'
