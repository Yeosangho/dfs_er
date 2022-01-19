#!/usr/bin/env python3

import numpy as np

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.acktr.policies import CnnPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

import constants as cnst


import gym

#from baselines.acktr.replaybuffer import ReplayBuffer_Custom, PrioritizedReplayBuffer_Custom

from collections import deque

import time
import threading
from PIL import Image
import math
import csv
from datetime import datetime
import gc
import cv2
from tqdm import tqdm

import sys
import argparse


def rgb2gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def process_frame(frame):
    floatframe = np.expand_dims(cv2.resize(rgb2gray(frame), (84, 84)), axis=0)
    return floatframe.astype(dtype=np.uint8)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype=np.uint8)
    data = process_frame(data)
    return data


def openLog(directory, filename, rlist):
    createTime = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
    with open(directory + str(createTime) + filename + '.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(rlist)
        myfile.close()
    return str(createTime) + filename


def writeLog(directory, filename, rlist):
    with open(directory + filename + '.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(rlist)
        myfile.close()


def getEpisodeListFrame(episodeList):
    i = f = None
    episode_list_count = 0
    whole_frame = 0
    for _ in range(episodeList.__len__()):
        episode = episodeList[episode_list_count]
        episodeEnd = False
        i, f = goNextEpisode(i, f, episode)
        score = 0
        while not episodeEnd:
            _, _, _, episodeEnd = check_score(i, f, episode)
            if (episodeEnd):
                break
            i += 1
            whole_frame += 1
        episode_list_count += 1
    return whole_frame


def step(i, f, episode):
    line = f.readline()
    traj = line[:-1].split(",")
    episodeEnd = False
    if not line:
        episodeEnd = True
        f.close()
        return '', 0.0, '', '', episodeEnd
    filename = ""
    for i in range(6 - len(str(episode))):
        filename += "0"

    filename += str(episode)


    imagefilename = ""
    for i in range(7-len(str(traj[0]))):
        imagefilename += "0"
    imagefilename += str(traj[0])

    #state = load_image(Config().SCREEN_PATH + gameName + "/" + str(episode) + "/" + imagefilename + ".png")
    state = load_image(Config().SCREEN_PATH  + filename + "/" + imagefilename + ".png")
    # print(screenpath + gameName + "/" + str(episode) + "/" + str(i) + ".png")
    # print(traj)
    # print(i)
    if ("False" in traj[3]):
        done = False
    else:
        done = True
    # done = bool(int(traj[3]))

    action = int(traj[4])
    translatedAction = actionTranslator[action]
    if(float(traj[2]) > float(Config().MAX_EP_SCORE_CUTOFF)):
        return state, float(traj[1]), done, translatedAction, True
    else:
        return state, float(traj[1]), done, translatedAction, episodeEnd


def check_score(i, f, episode):
    line = f.readline()
    traj = line[:-1].split(",")
    episodeEnd = False
    if not line:
        episodeEnd = True
        f.close()
        return 0.0, '', '', episodeEnd
    # print(screenpath + gameName + "/" + str(episode) + "/" + str(i) + ".png")
    # print(traj)
    # print(i)

    if ( "False" in traj[3]):
        done = False
    else:
        done = True
    # done = bool(int(traj[3]))

    action = int(traj[4])
    translatedAction = actionTranslator[action]
    return float(traj[1]), done, translatedAction, episodeEnd


def goNextEpisode(count, file, episode):
    count = 0
    filename = ""
    for i in range(6 - len(str(episode))):
        filename += "0"

    filename += str(episode)
    file = open(Config().TRAJ_PATH + filename + ".csv", 'r')
    file.readline()
    return count, file


def set_n_step(container, n, ts):
    # print(container)
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    #n_step_reward = sum([t[2] * Config.GAMMA ** i for i, t in enumerate(t_list[0:min(len(t_list), n)])])
    #
    #for begin in range(len(t_list)):
    #    end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
    #    # extend[n_reward, n_next_s, n_done, actual_n]
    #    t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end - begin + 1, ts])
    #    n_step_reward = (n_step_reward - t_list[begin][2]) / Config.GAMMA
    return t_list


def actionTranslate(gymActions, dataSetActions):
    actionTranslation = []
    length = 0
    for action in dataSetActions:
        i = 0
        for gymAction in gymActions:
            if (action == gymAction):
                actionTranslation.append(i)
            i = i + 1
        if (length == actionTranslation.__len__()):
            actionTranslation.append(0)
        length = actionTranslation.__len__()
    return actionTranslation




def get_goodepisode(episodeList, percent):
    episodeEnd = False
    i = f = None
    epsidoe_list_count = 0

    score_dtype = [('key', int), ('score', float), ('freq', object)]
    ep_score_array = np.zeros(episodeList.__len__(), dtype=score_dtype)

    action_repetition = 0


    episode_action_counts = []

    for _ in range(episodeList.__len__()):
        episode = episodeList[epsidoe_list_count]
        episode = episode.split('.')[0]
        action_counts = []
        oldAction = None
        episodeEnd = False
        i, f = goNextEpisode(i, f, episode)
        score = 0
        while not episodeEnd:
            reward, _, action, episodeEnd = check_score(i, f, episode)

            if(action == oldAction and oldAction is not None):
                action_repetition += 1
            elif(action != oldAction and oldAction is not None):
                for j in range(action_repetition+1):
                    action_counts.append(action_repetition+1 - j)
                action_repetition = 0
            oldAction = action
            i += 1
            score += reward

        #episode_action_counts.append(action_counts)
        ep_score_array[epsidoe_list_count] = (int(episode), -score, action_counts)
        epsidoe_list_count += 1
    ep_score_array = np.sort(ep_score_array, order='score')


    #print(ep_score_array)
    print((ep_score_array['key'][:int(episodeList.__len__() * percent)]))
    print(ep_score_array[0])
    print(ep_score_array[0][2].__len__())
    return ep_score_array


def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def sign(x): return 1 if x >= 0 else -1


'''
class Trainer():
    def __init__(self, name, env, agent, episodeList, sample_log, replay_log, delete_log, episode_log,
                 epochscore_log, LIFE_RESET, ACTOR_ITR, HUMAN_ITR, REWARD_CLIP, USE_DFS):
        self.name = name
        self.env = env
        self.agent = agent
        self.episodeList = episodeList
        self.episode = self.i = self.f = self.episode_freq = None
        self.LIFE_RESET = LIFE_RESET
        self.actor_itr = ACTOR_ITR
        self.human_itr = HUMAN_ITR
        self.reward_clip = REWARD_CLIP
        self.use_dfs = USE_DFS
        self.update_freq = Config.UPDATE_FREQ

        self.score_per_epoch = 0
        self.episode_per_epoch = 0
        self.pre_score_per_epoch =0

    def init_pretrain(self):


        scores, e, replay_full_episode = [], 0, None

        # random.shuffle(self.episodeList)
        epsidoe_list_count = 0
        self.episode = str(self.episodeList[epsidoe_list_count][0])
        episode_count = 0
        old_episode_count = 0
        self.episode_freq = self.episodeList[epsidoe_list_count][2]

        self.i, self.f = goNextEpisode(self.i, self.f, self.episode)
        episodeEnd = False





        actor_done, actor_score, actor_n_step_reward, actor_state = False, 0, None, self.env.reset()
        human_done, human_score, human_n_step_reward = False, 0, None
        human_state, _, _, _, episodeEnd = step(self.i, self.f, self.episode)
        self.i = self.i + 1
        t_q_actor = deque(maxlen=4)
        t_q_human = deque(maxlen=4)
        if(self.use_dfs):
            t_q_human = deque(maxlen=Config.ACTION_FREQ_RANGE)



        actor_state = process_frame(actor_state)



        human_episode = []
        human_ep_count = 0

        ##Store the Human Data Until PRETRAIN_START
        while (self.agent.initPretrain() or human_ep_count == 0):

            while episodeEnd is False:
                startTime = time.time()

                reward = 0

                freq = self.episode_freq[episode_count]


                if (freq > Config.ACTION_FREQ_COND):
                    #episode_count += int(Config.ACTION_FREQ_RANGE/10)
                    #episode_count += Config.ACTION_FREQ_RANGE
                    episode_count += Config.SKIP_HUMAN
                elif(episode_count < self.episode_freq.__len__()):
                    episode_count += Config.SKIP_HUMAN

                for _ in range(episode_count - old_episode_count):
                    if(episodeEnd != True):
                        next_state, sub_reward, human_done, action, episodeEnd = step(self.i, self.f, self.episode)
                        self.i = self.i + 1
                        reward += sub_reward
                        if(episodeEnd == True):
                            next_state, reward, human_done, action = np.copy(old_next_state), old_reward, old_human_done, old_action
                        else :
                            old_next_state, old_reward, old_human_done, old_action = np.copy(next_state), reward, human_done, action


                if(freq>Config.ACTION_FREQ_COND and self.use_dfs):
                    action += self.agent.actions




                if(episodeEnd == True):
                    break

                old_episode_count = episode_count


                human_score += reward


                log_scale_reward = sign(reward) * math.log(1+ abs(reward))
                t_q_human.append(
                    [action, np.copy(log_scale_reward), np.copy(next_state), human_done, 1.0])
                if(self.use_dfs):
                    if len(t_q_human) == t_q_human.maxlen:
                        t_list = list(t_q_human)
                        r1_reward = sum([t[1] for i, t in enumerate(t_list[t_list.__len__()-4:t_list.__len__()])])
                        r2_reward = sum([t[1] for i, t in enumerate(t_q_human)]) / Config.NORMALIZE_REWARD
                        if(self.reward_clip==True):
                           if(r2_reward>0):
                               r2_reward = 1.0
                           else:
                               r2_reward = 0.0
                           if(human_done):
                               r2_reward = -1.0

                           if(r1_reward>0):
                               r1_reward = 1.0
                           else:
                               r1_reward = 0.0
                           if(human_done):
                               r1_reward = -1.0

                        else :
                           r2_reward = sign(r2_reward) * math.log(1 + abs(r2_reward)) if not human_done else sign(-100) * math.log(
                               1 + abs(-100))
                           r1_reward = sign(r1_reward) * math.log(1 + abs(r1_reward)) if not human_done else sign(-100) * math.log(
                               1 + abs(-100))
                        t_q_human[t_q_human.__len__() -1].extend([r1_reward, episode_count, r2_reward])
                        human_episode.append(t_q_human[t_q_human.__len__() -1])
                else:
                    if len(t_q_human) == t_q_human.maxlen:
                        r1_reward = sum([t[1] for i, t in enumerate(t_q_human)])
                        if(self.reward_clip==True):
                           if(r1_reward>0):
                               r1_reward = 1.0
                           else:
                               r1_reward = 0.0
                           if(human_done):
                               r1_reward = -1.0
                        else :
                            if(USE_MODEL == 2):
                                #r1_reward = r1_reward
                                r1_reward = sign(r1_reward) * math.log(1 + abs(r1_reward)) if not human_done else sign(-100) * math.log( 1 + abs(-100))
                            else :
                                r1_reward = sign(r1_reward) * math.log(1 + abs(r1_reward)) if not human_done else sign(-100) * math.log(
                               1 + abs(-100))


                        t_q_human[t_q_human.__len__() -1].extend([r1_reward, episode_count])
                        human_episode.append(t_q_human[t_q_human.__len__() -1])

                human_state = next_state


            if (episodeEnd):
                # handle transitions left in t_q

                print("human : episode end")

                for i in range(human_episode.__len__()):

                    if(not self.agent.isHumanFull()):
                        self.agent.perceive(human_episode[i], human_score)
                epsidoe_list_count += 1
                if (epsidoe_list_count == self.episodeList.__len__()):
                    # random.shuffle(self.episodeList)
                    epsidoe_list_count = 0
                    self.episode = str(self.episodeList[epsidoe_list_count][0])
                    self.episode_freq = self.episodeList[epsidoe_list_count][2]
                else:
                    self.episode = str(self.episodeList[epsidoe_list_count][0])
                    self.episode_freq = self.episodeList[epsidoe_list_count][2]
                episode_count = old_episode_count=0
                self.i, self.f = goNextEpisode(self.i, self.f, self.episode)
                human_done, human_score, human_n_step_reward = False, 0, None

                human_state, _, _, _, episodeEnd = step(self.i, self.f, self.episode)
                self.i = self.i + 1


                t_q_human = deque(maxlen=4)
                if (self.use_dfs):
                    t_q_human = deque(maxlen=Config.ACTION_FREQ_RANGE)
                human_episode = []
                human_ep_count += 1
'''

def train(params):
    policy_fn = CnnPolicy

    dataflow_config = {
        'DFS-ER':True,
        'r2_cond': 110,
        'r2_skip': 12,  # 처음에는 110/12 로했고, 지금(4/7) 은 110/20 으로 테스트중
        'n_step': 10,
        'future_rewards': True,             # Should return future discounted rewards?
        'exclude_zero_actions': False,      # Should exclude zero actions
        'remap_actions': False,             # Should remap to smaller action set?
        'clip_rewards': True,               # Clip rewards to [-1, 1]
        'monte-specific-blackout': True,    # Cover up score and lives indicators
        'pong-specific-blackout': False,    # Cover up scores in pong
        'gamma': params.gamma,              # reward discount factor
        'frame_history': 4,                 # What is minimum number of expert frames since beginning of episode?
        'frameskip': 4,                     # frameskip
        'preload_images': True,             # Preload images from hard drive or keep reloading ?
        'gdrive_data_id': cnst.MONTE_DATA_GDRIVE_ID,
        'data_dir': cnst.DATA_DIR,
        'img_dir': cnst.MIKE_IMG_DIR,
        'traj_dir': cnst.MIKE_TRAJECTORIES_DIR,
        'stat_dir': cnst.MIKE_STATES_DIR,
        'batch_size': params.expert_nbatch,
        'max_score_cutoff': params.exp_max_score,  # What is maximum expert score we can show? Used to cut expert data
        'min_score_cutoff': 20000,                 # What is minimum score to count trajectory as expert
        'process_lost_lifes': True,                # Should loss of life zero future discounted reward?
        'use_n_trajectories': params.use_n_trajectories if 'use_n_trajectories' in params else None
    }

    the_seed = np.random.randint(10000)
    print(80 * "SEED")
    print("Today's lucky seed is {}".format(the_seed))
    print(80 * "SEED")

    env = VecFrameStack(
        make_atari_env(
            env_id=params.env,
            num_env=params.num_env,
            seed=the_seed,
            limit_len=params.limit_len,
            limit_penalty=params.limit_penalty,
            death_penalty=params.death_penalty,
            step_penalty=params.step_penalty,
            random_state_reset=params.random_state_reset,
            dataflow_config=dataflow_config
        ),
        params.frame_stack
    )

    learn(
        policy=policy_fn,
        env=env,
        seed=the_seed,
        params=params,
        dataflow_config=dataflow_config,
        expert_nbatch=params.expert_nbatch,
        exp_adv_est=params.exp_adv_est,
        load_model=params.load_model,
        gamma=params.gamma,
        nprocs=params.num_env,
        nsteps=params.nsteps,
        ent_coef=params.ent_coef,
        expert_coeff=params.exp_coeff,
        lr=params.lr,
        lrschedule=params.lrschedule,
    )

    env.close()


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params = atari_arg_parser().parse_args()
    logger.configure(dir=cnst.openai_logdir())
    #human_memory = ReplayBuffer_Custom(HUMAN_MEMORY_SIZE, True, REWARD_CLIP, env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)

    train(params)


if __name__ == '__main__':
    main()
