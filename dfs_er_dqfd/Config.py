# -*- coding: utf-8 -*-
import os


class Config:
    #ENV_NAME = 'QbertNoFrameskip-v0'
    #ENV_NAME =  'MsPacmanNoFrameskip-v0'
    #ENV_NAME = 'SpaceInvadersNoFrameskip-v0'
    ENV_NAME = 'MontezumaRevengeNoFrameskip-v0'


    #GAME_NAME = 'qbert'
    #GAME_NAME = "mspacman"
    #GAME_NAME = "spaceinvaders"
    GAME_NAME = "revenge"


    ACTION_SET = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
                  'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE',
                   'DOWNLEFTFIRE']

    MAX_EP_SCORE_CUTOFF = 30000
    UPDATE_FREQ = 4
    NAC_ALPHA = 0.1
    SKIP_EPISODE = True
    MAX_EP_LEN = 50000
    GAME_RESET_LIFE_LOSS = False
    ACTION_FREQ_COND = 70
    ACTION_FREQ_RANGE = 12
    #NORMALIZE_REWARD = int(ACTION_FREQ_RANGE/4)
    #NORMALIZE_REWARD = int(ACTION_FREQ_RANGE/4)
    NORMALIZE_REWARD = 1
    EPOCH = 100000
    SKIP_R1 = 1
    SKIP_HUMAN = 4
    N_STEP = 10
    ALPHA = 0.4
    BETA = 0.6
    HUMAN_EPSILON = 0.001
    ACTOR_EPSILON = 0.001
    FINAL_HUMAN_RATIO = 0.03
    HISTORY_LENGTH = 4
    EXP_LENGTH = 1
    LOG_FREQ = 1000
    TEST_EP = 10
    START_STEP = 0

    TRAJ_PATH = "/home/soboru963/atari_v1/trajectories/"
    SCREEN_PATH = '/home/soboru963/atari_v1/screens/'

    #TRAJ_PATH = "/home/soboru963/Downloads/our-data/trajectories/"
    #SCREEN_PATH = "/home/soboru963/Downloads/our-data/screens/"

    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 0.01 # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.999
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 32  # size of minibatch
    UPDATE_TARGET_NET = 10000  # update eval_network params every 200 steps

    #UPDATE_TARGET_NET = 1  # update eval_network params every 200 steps
    LEARNING_RATE = 0.0001
    LEARNING_RATE_FINAL = 0.0001
    LEARNING_RATE_DECAY_STEP = 500000
    #LAMBDA = [1.0, 1.0, 1.0, 1e-5]
    #LAMBDA = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    LAMBDA = [1.0, 1.0, 1.0, 1e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model')
    ACTOR_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actor/')
    LEARNER_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learner/')
    GIF_STEP = 10

    trajectory_n = 10  # for n-step TD-loss (both demo data and generated data)

class DDQNConfig(Config):
    demo_mode = 'get_demo'


class DQfDConfig(Config):
    demo_mode = 'use_demo'
