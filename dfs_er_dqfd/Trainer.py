# -*- coding: utf-8 -*

import gym

from Config import Config, DQfDConfig
from DQfD import DQfD
from replaybuffer import ReplayBuffer_Custom, PrioritizedReplayBuffer_Custom
from singlereplaybuffer import SingleReplayBuffer_Custom
from ReplayMemoryManager import ReplayMemoryManager

from collections import deque

import time
from helper import *
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

'''
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
    state = load_image(Config().SCREEN_PATH + gameName + "/" + str(episode) + "/" + str(traj[0]) + ".png")
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
    file = open(Config().TRAJ_PATH + gameName + "/" + str(episode) + ".txt", 'r')
    file.readline()
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


class Trainer():
    def __init__(self, name, env, agent, episodeList, sample_log, replay_log, delete_log, episode_log,
                 epochscore_log, LIFE_RESET, ACTOR_ITR, HUMAN_ITR, REWARD_CLIP, USE_DFS):
        self.name = name
        self.env = env
        self.agent = agent
        self.episodeList = episodeList
        self.episode = self.i = self.f = self.episode_freq = None
        self.sample_log = sample_log
        self.replay_log = replay_log
        self.delete_log = delete_log
        self.episode_log = episode_log
        self.epochscore_log = epochscore_log
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

    def pretrain(self):
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

        actor_history = np.zeros([Config.HISTORY_LENGTH, 84, 84], dtype=np.uint8)


        actor_state = process_frame(actor_state)

        for _ in range(Config.HISTORY_LENGTH):

            actor_history[:-1] = actor_history[1:]
            actor_history[-1] = actor_state

        human_episode = []
        episode_frames = []
        human_ep_count = 0

        frame_count =0
        actor_ep_count = 0
        pre_life =0
        for time_step in tqdm(range(self.agent.pretrain_step), ncols=70, initial=0):
            startTime = time.time()
            # if(train_itr % actor_human_ratio != 0 ):
            action = self.agent.egreedy_action(actor_history)  # e-greedy action for train
            env_action = 0
            if (action >= self.agent.actions):
                env_action = action - self.agent.actions

                e_freq = Config.ACTION_FREQ_RANGE
            elif (action < self.agent.actions):
                env_action = action
                e_freq = 4

            reward = 0

            for i in range(e_freq):
                if (actor_done == True):
                    next_state, sub_reward, _, info = self.env.step(env_action)
                else:
                    next_state, sub_reward, actor_done, info = self.env.step(env_action)
                # env.render()
                reward += sub_reward
            next_state = process_frame(next_state)
            frame_count += 1
            # print(next_state)
            actor_history[:-1] = actor_history[1:]
            actor_history[-1] = next_state[0]

            if (info.get('ale.lives') < pre_life):
                if (self.LIFE_RESET == True and actor_done == False):
                    actor_done = True
                elif (Config.GAME_RESET_LIFE_LOSS):
                    life_loss = True
            else:
                life_loss = False

            pre_life = info.get('ale.lives')

            actor_score += reward
            if (self.reward_clip == True):
                if (reward > 0):
                    reward = 1.0
                else:
                    reward = 0.0
                if (actor_done or life_loss):
                    reward = -1.0
            else:
                reward = sign(reward) * math.log(1 + abs(reward)) if not (
                            actor_done or life_loss) else sign(-100) * math.log(
                    1 + abs(-100))
            reward_to_sub = 0. if len(t_q_actor) < t_q_actor.maxlen else t_q_actor[0][
                2]  # record the earliest reward for the sub
            t_q_actor.append([action, reward, np.copy(next_state), actor_done, 0.0, life_loss])

            if (frame_count > Config.MAX_EP_LEN and Config.SKIP_EPISODE):
                actor_done = True


            self.agent.train_Q_network_with_update_freq()  # train along with generation




            if self.agent.getTimeStep() % Config().UPDATE_TARGET_NET == 0:
                # print("actor_update_target"+str(train_itr))
                #self.agent.save_model()
                self.agent.sess.run(self.agent.update_target_net)

                if (self.agent.getTimeStep() % Config().EPOCH == 0):
                    if(self.episode_per_epoch == 0):
                        epoch_score = self.pre_score_per_epoch
                    else:
                        epoch_score = self.score_per_epoch / self.episode_per_epoch
                        self.pre_score_per_epoch = epoch_score
                        self.score_per_epoch = 0
                        self.episode_per_epoch = 0


                    writeLog(Config.ACTOR_DATA_PATH + 'epochscore/', self.epochscore_log,
                                 [str(self.agent.getTimeStep() ), str(epoch_score), str(0)])


            if (self.agent.getTimeStep() % 10000 == 1):
                print("process time : " + str(time.time() - startTime) )

            if actor_done:
                # handle transitions left in t_q
                print("actor end")


                frame_count = 0
                self.score_per_epoch += actor_score
                self.episode_per_epoch += 1
                actor_done, actor_score, actor_n_step_reward, actor_state = False, 0, None, self.env.reset()
                actor_history = np.zeros([Config.HISTORY_LENGTH, 84, 84], dtype=np.uint8)
                actor_state = process_frame(actor_state)
                pre_life = 0
                for _ in range(Config.HISTORY_LENGTH):
                    actor_history[:-1] = actor_history[1:]
                    actor_history[-1] = actor_state

                t_q_actor = deque(maxlen=4)

                actor_ep_count += 1



    def init_run(self):


        episode_frames = []

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
        t_q_actor = deque(maxlen=1)
        t_q_human = deque(maxlen=4)
        if(self.use_dfs):
            t_q_human = deque(maxlen=Config.ACTION_FREQ_RANGE)

        actor_history = np.zeros([Config.HISTORY_LENGTH, 84, 84], dtype=np.uint8)


        frame_count = 0
        actor_state = process_frame(actor_state)

        for _ in range(Config.HISTORY_LENGTH):
            actor_history[:-1] = actor_history[1:]
            actor_history[-1] = actor_state

        actor_episode = []
        human_episode = []
        life_loss = False
        pre_life = 0
        actor_ep_count = 0
        while (self.agent.initRL() or actor_ep_count == 0):

            while actor_done is False and episodeEnd is False:
                startTime = time.time()
                # if(train_itr % actor_human_ratio != 0 ):
                for _ in range(self.actor_itr):
                    action = self.agent.egreedy_action(actor_history)  # e-greedy action for train
                    env_action = 0
                    if (action >= self.agent.actions):
                        env_action = action - self.agent.actions

                        e_freq = Config.ACTION_FREQ_RANGE
                    elif (action < self.agent.actions):
                        env_action = action
                        e_freq = 4

                    reward = 0

                    for i in range(e_freq):
                        if(actor_done == True):
                            next_state, sub_reward, _, info = self.env.step(env_action)
                        else:
                            next_state, sub_reward, actor_done, info = self.env.step(env_action)
                        # env.render()
                        reward += sub_reward
                    next_state = process_frame(next_state)
                    frame_count += 1
                    # print(next_state)
                    actor_history[:-1] = actor_history[1:]
                    actor_history[-1] = next_state[0]


                    if (info.get('ale.lives') < pre_life ):
                        if (self.LIFE_RESET == True and actor_done == False):
                            actor_done = True
                        elif (Config.GAME_RESET_LIFE_LOSS):
                            life_loss = True
                    else :
                        life_loss = False

                    pre_life = info.get('ale.lives')

                    actor_score += reward
                    if (self.reward_clip == True):
                        if (reward > 0):
                            reward = 1.0
                        else:
                            reward = 0.0
                        if (actor_done or life_loss):
                            reward = -1.0
                    else:
                        reward = sign(reward) * math.log(1 + abs(reward)) if not (actor_done or life_loss) else sign(-100) * math.log(
                            1 + abs(-100))
                    reward_to_sub = 0. if len(t_q_actor) < t_q_actor.maxlen else t_q_actor[0][
                        2]  # record the earliest reward for the sub
                    t_q_actor.append([action, reward, np.copy(next_state), actor_done, 0.0, life_loss])
                    self.agent.perceive(t_q_actor[0], actor_score)

                    if (frame_count > Config.MAX_EP_LEN and Config.SKIP_EPISODE):
                        actor_done = True




                    actor_state = next_state


            if actor_done:
                # handle transitions left in t_q
                print("actor end")


                frame_count = 0
                actor_done, actor_score, actor_n_step_reward, actor_state = False, 0, None, self.env.reset()
                actor_history = np.zeros([Config.HISTORY_LENGTH, 84, 84], dtype=np.uint8)
                actor_state = process_frame(actor_state)
                pre_life = 0
                for _ in range(Config.HISTORY_LENGTH):
                    actor_history[:-1] = actor_history[1:]
                    actor_history[-1] = actor_state

                t_q_actor = deque(maxlen=1)

                episode_frames = []
                actor_episode = []
                actor_ep_count += 1

            e += 1
    def run(self):

        episode_frames = []
        episode_score = []

        count = 0
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
        t_q_actor = deque(maxlen=1)
        t_q_human = deque(maxlen=4)
        if(self.use_dfs):
            t_q_human = deque(maxlen=Config.ACTION_FREQ_RANGE)

        actor_history = np.zeros([Config.HISTORY_LENGTH, 84, 84], dtype=np.uint8)

        episode_idx = 0
        frame_count = 0
        actor_state = process_frame(actor_state)

        for _ in range(Config.HISTORY_LENGTH):
            actor_history[:-1] = actor_history[1:]
            actor_history[-1] = actor_state

        actor_episode = []
        human_episode = []

        pre_life = 0
        life_loss = False


        for time_step in tqdm(range(self.agent.rl_step), ncols=70, initial=0):

            startTime = time.time()
            # if(train_itr % actor_human_ratio != 0 ):
            action = self.agent.egreedy_action(actor_history)  # e-greedy action for train
            env_action= 0
            if (action >= self.agent.actions):
                env_action = action - self.agent.actions
                e_freq = Config.ACTION_FREQ_RANGE
            elif (action < self.agent.actions):
                env_action = action
                e_freq = 4

            reward = 0

            for _ in range(e_freq):

                if(actor_done == True):
                    next_state, sub_reward, _, info = self.env.step(env_action)
                else:
                    next_state, sub_reward, actor_done, info = self.env.step(env_action)

                reward += sub_reward
            frame_count += 1
                # env.render()
            actor_score += reward
            next_state = process_frame(next_state)


            # print(next_state)
            actor_history[:-1] = actor_history[1:]
            actor_history[-1] = next_state[0]

            if (info.get('ale.lives') < pre_life ):
                if (self.LIFE_RESET == True and actor_done == False):
                    actor_done = True
                elif(Config.GAME_RESET_LIFE_LOSS) :
                    life_loss = True
            else :
                life_loss = False

            pre_life = info.get('ale.lives')

            if (self.reward_clip == True):
                if (reward > 0):
                    reward = 1.0
                else:
                    reward = 0.0
                if (actor_done or life_loss):
                    reward = -1.0
            else:
                reward = sign(reward) * math.log(1 + abs(reward)) if not (actor_done or life_loss) else sign(-100) * math.log(
                    1 + abs(-100))

            reward_to_sub = 0. if len(t_q_actor) < t_q_actor.maxlen else t_q_actor[0][
                2]  # record the earliest reward for the sub
            t_q_actor.append([ action, reward, np.copy(next_state), actor_done, 0.0, life_loss])
            self.agent.perceive(t_q_actor[0], actor_score)

            if (frame_count > Config.MAX_EP_LEN and Config.SKIP_EPISODE):
                actor_done = True

            actor_state = next_state
            # if (train_itr % actor_human_ratio == 0):

            self.agent.train_Q_network_with_update_freq()  # train along with generation


            replay_full_episode = replay_full_episode or e
            if self.agent.getTimeStep() % Config().UPDATE_TARGET_NET == 0:
                # print("actor_update_target"+str(train_itr))
                #self.agent.save_model()
                self.agent.sess.run(self.agent.update_target_net)

                if(self.agent.getTimeStep() % Config().EPOCH == 0):
                    sum_score = 0.0

                    epoch_score = 0
                    if(self.episode_per_epoch == 0):
                        epoch_score = self.pre_score_per_epoch
                    else:
                        epoch_score = self.score_per_epoch / self.episode_per_epoch
                        self.pre_score_per_epoch = epoch_score
                        self.score_per_epoch = 0
                        self.episode_per_epoch = 0

                    writeLog(Config.ACTOR_DATA_PATH + 'epochscore/', self.epochscore_log,
                             [str(self.agent.getTimeStep() ),str(epoch_score), str(0)])

            if (self.agent.getTimeStep() % 10000 == 1):
                print("process time : " + str(time.time() - startTime))

            if actor_done:
                # handle transitions left in t_q
                print("actor end")


                print("episode: {}   score: {}    epsilon: {}"
                      .format(episode_idx, actor_score,
                              self.agent.epsilon))
                episode_score.append(actor_score)
                writeLog(Config.ACTOR_DATA_PATH + 'episodescore/', self.episode_log,
                         [str(episode_idx), str(actor_score), str(self.agent.time_step)])

                    # 주기적으로 에피소드의 gif 를 저장하고, 모델 파라미터와 요약 통계량을 저장한다.
                # if episode_count % Config.GIF_STEP == 0 and episode_count != 0 :
                #    time_per_step = 0.05
                #    images = np.array(episode_frames)
                #    make_gif(images, './frames/dqfd_image' + str(episode_count) + '.gif',
                #             duration=len(images) * time_per_step, true_image=True, salience=False)
                frame_count = 0
                self.episode_per_epoch += 1
                self.score_per_epoch += actor_score
                actor_done, actor_score, actor_n_step_reward, actor_state = False, 0,None, self.env.reset()
                actor_history = np.zeros([Config.HISTORY_LENGTH, 84, 84], dtype=np.uint8)
                actor_state = process_frame(actor_state)
                pre_life = 0
                for _ in range(Config.HISTORY_LENGTH):
                    actor_history[:-1] = actor_history[1:]
                    actor_history[-1] = actor_state

                t_q_actor = deque(maxlen=1)

                episode_idx = episode_idx + 1





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQfD Sampling Scheduling Test')
    parser.add_argument('--testname',
                        default='',
                        help="Testname")
    parser.add_argument('--model',
                        type=int,
                        default=1,
                        help="0 -> DDQN with n step q loss, 1 -> DQfD, 2->NAC")
    parser.add_argument('--memory',
                        type=int,
                        default=0,
                        help="0 -> Normal Replay buffer, 1 -> Priority Replay buffer 2 -> single replay buffer")
    parser.add_argument('--humansize',
                        type=int,
                        default=50000,
                        help="size of human data")
    parser.add_argument('--actorsize',
                        type=int,
                        default=50000,
                        help="size of actor data")
    parser.add_argument('--rlstep',
                        type=int,
                        default=3000000,
                        help="how many train the model by merged data")
    parser.add_argument('--pretrainstep',
                        type=int,
                        default=750000,
                        help="how many pretrain the model by human data")
    parser.add_argument('--sampleschedule',
                        type=int,
                        default=5,
                        help="0 -> linear, 1 -> logarithm_fast(human exponential decay, decay step 5000) 2 -> logarithm(human exponential decay step 10000) 3 -> constant(0.75) 4 -> only actor, 5 -> mimic single replay management")
    parser.add_argument('--sampleschedulelength',
                        type=int,
                        default=1000000,
                        help="how long decay the sampling ratio")
    parser.add_argument('--rlstart',
                        type=int,
                        default=5000,
                        help=" Memory Buffer Size that Reinforcement Learning Start from")
    parser.add_argument('--pretrainstart',
                        type=int,
                        default=50000,
                        help="Memory Buffer Size that Pretrain  Start from")
    parser.add_argument('--lossmod',
                        type=int,
                        default=0,
                        help="Use Loss Model Modification Version 0(False) 1(True)")
    parser.add_argument('--lifereset',
                        type=int,
                        default=0,
                        help="Use Life Reset Mode 0(False) 1(True)")
    parser.add_argument('--rewardclip',
                        type=int,
                        default=0,
                        help="Use reward clipping 0(False) 1(True)")
    parser.add_argument('--optimizer',
                        type=int,
                        default=0,
                        help="Use optimizer adam 0 rmsprop 1")
    parser.add_argument('--lossclip',
                        type=int,
                        default=1,
                        help="Use lossclip 0(False) 1(True)")
    parser.add_argument('--usedfs',
                        type=int,
                        default=0,
                        help="Use usedfs 0(False) 1(True)")
    parser.add_argument('--dynamic_nq',
                        type=int,
                        default=1,
                        help="Use dynamic n q reward 0(False) 1(True)")

    args = parser.parse_args()
    testname = args.testname
    model = args.model
    memory = args.memory
    humansize = args.humansize
    actorsize = args.actorsize
    rlstep = args.rlstep + 1
    pretrainstep = args.pretrainstep + 1
    sampleschedule = args.sampleschedule
    sampleschedulelength = args.sampleschedulelength
    rlstart = args.rlstart
    pretrainstart = args.pretrainstart
    optimizer = args.optimizer
    dynamic_nq = args.dynamic_nq

    lossmod = args.lossmod
    lifereset = args.lifereset
    rewardclip = args.rewardclip
    lossclip = args.lossclip
    usedfs = args.usedfs

    USE_MODEL = model
    #if model == 1 : USE_DQfD = True
    #elif model == 0 : USE_DQfD = False

    USE_PRIORITY_REPLAY = False
    USE_SINGLE_REPLAY = False
    if memory == 1 : USE_PRIORITY_REPLAY = True
    elif memory == 0 : USE_PRIORITY_REPLAY = False
    elif memory == 2 : USE_SINGLE_REPLAY = True

    HUMAN_MEMORY_SIZE = humansize
    ACTOR_MEMORY_SIZE = actorsize
    RL_STEP = rlstep
    PRETRAIN_STEP = pretrainstep
    RL_START = rlstart
    PRETRAIN_START = pretrainstart

    USE_LINEAR = False
    USE_LOGARITHM_FAST = False
    USE_LOGARITHM = False
    USE_CONSTANT = False
    USE_ONLY_ACTOR = False
    USE_MIMIC_SINGLE_REPLAY = False
    if sampleschedule == 0 : USE_LINEAR = True
    elif sampleschedule == 1 : USE_LOGARITHM_FAST = True
    elif sampleschedule == 2: USE_LOGARITHM = True
    elif sampleschedule == 3: USE_CONSTANT = True
    elif sampleschedule == 4: USE_ONLY_ACTOR = True
    elif sampleschedule == 5 : USE_MIMIC_SINGLE_REPLAY = True
    DECAY_LENGTH = sampleschedulelength

    LOSS_MOD = False
    if(lossmod == 1): LOSS_MOD = True

    LIFE_RESET = False
    if(lifereset == 1): LIFE_RESET = True

    ACTOR_ITR = 1
    HUMAN_ITR = 1
    if(USE_ONLY_ACTOR == True):
        HUMAN_ITR = 0

    REWARD_CLIP = False
    if(rewardclip==1):
        REWARD_CLIP = True
    LOSSCLIP = False
    if(lossclip == 1):
        LOSSCLIP = True
    OPTIMIZER = optimizer

    USE_DFS = False
    if(usedfs == 1):
        USE_DFS =True

    USE_DYNAMIC_N_Q = False
    if(dynamic_nq == 1):
        USE_DYNAMIC_N_Q = True

    session = tf.InteractiveSession()
    env = gym.make(Config.ENV_NAME)


    scores, e, replay_full_episode = [], 0, None
    gameName = Config.GAME_NAME
    gameID = Config.ENV_NAME
    dataSetAction = Config.ACTION_SET
    env = gym.make(gameID)
    gymAction = env.unwrapped.get_action_meanings()
    actionTranslator = actionTranslate(gymAction, dataSetAction)
    #episodeList = os.listdir(Config().TRAJ_PATH)  # dir is your directory path
    episodeList = os.listdir(Config().TRAJ_PATH + gameName + '/')  # dir is your directory path
    demo_size = 0

    if(USE_SINGLE_REPLAY):
        memory = SingleReplayBuffer_Custom(HUMAN_MEMORY_SIZE, ACTOR_MEMORY_SIZE, False, REWARD_CLIP, env.action_space.n, USE_DFS,
                                           USE_DYNAMIC_N_Q)
        human_memory = actor_memory = memory
    else:
        if (USE_PRIORITY_REPLAY):
            human_memory = PrioritizedReplayBuffer_Custom(HUMAN_MEMORY_SIZE, Config.ALPHA, True, REWARD_CLIP, env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)
            actor_memory = PrioritizedReplayBuffer_Custom(ACTOR_MEMORY_SIZE, Config.ALPHA, False, REWARD_CLIP, env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)
        else:
            human_memory = ReplayBuffer_Custom(HUMAN_MEMORY_SIZE, True, REWARD_CLIP, env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)
            actor_memory = ReplayBuffer_Custom(ACTOR_MEMORY_SIZE, False, REWARD_CLIP, env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)

    replay_manager = ReplayMemoryManager(human_memory, actor_memory, DECAY_LENGTH, USE_LINEAR, USE_LOGARITHM_FAST,
                                         USE_LOGARITHM, USE_CONSTANT, USE_ONLY_ACTOR, USE_MIMIC_SINGLE_REPLAY, USE_PRIORITY_REPLAY, USE_SINGLE_REPLAY)

    episodeList = get_goodepisode(episodeList, 0.20)

    #logfile initialize
    #######################################################################################################
    sample_log = openLog(Config.LEARNER_DATA_PATH + 'sampleexp/', '[' + str(testname) +  ']',
                         ['step', 'value', 'actor_value', 'human_value', 'age', 'demo', 'qvalue'])
    replay_log = openLog(Config.LEARNER_DATA_PATH + 'replaymemory/', '[' + str(testname) +  ']',
                         ['step', 'root_priority', 'root_ts', 'root_demo', 'alpha', 'beta'])
    delete_log = openLog(Config.ACTOR_DATA_PATH + 'deletedexp/', '[' + str(testname) + ']',
                         ['step', 'train_itr', 'value', 'age', 'demo'])
    episode_log = openLog(Config.ACTOR_DATA_PATH + 'episodescore/', '[' + str(testname) +  ']',
                          ['episode', 'score', 'train_step'])
    epochscore_log = openLog(Config.ACTOR_DATA_PATH + 'epochscore/', '[' + str(testname) +  ']',
                             ['train_step', 'score', 'average_ep', 'average_step'])
    #######################################################################################################

    screenpath = Config.SCREEN_PATH
    trajpath = Config.TRAJ_PATH

    agent = DQfD('learner', env, DQfDConfig(), session, replay_manager, USE_MODEL, USE_PRIORITY_REPLAY, PRETRAIN_STEP, RL_STEP, PRETRAIN_START, RL_START, LOSS_MOD, OPTIMIZER, LOSSCLIP, USE_DFS)
    env = gym.make(Config.ENV_NAME)

    trainer = Trainer('learner', env, agent, episodeList,  sample_log, replay_log, delete_log,
                      episode_log, epochscore_log, LIFE_RESET, ACTOR_ITR, HUMAN_ITR, REWARD_CLIP, USE_DFS)
    trainer.init_pretrain()
    trainer.pretrain()
    if(USE_ONLY_ACTOR == True):
        human_memory = None
        replay_manager.humanMemory = None
        gc.collect()
    print('init run')
    trainer.init_run()
    print('run')
    trainer.run()

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQfD Sampling Scheduling Test')
    parser.add_argument('--testname',
                        default='',
                        help="Testname")
    parser.add_argument('--model',
                        type=int,
                        default=1,
                        help="0 -> DDQN with n step q loss, 1 -> DQfD, 2->NAC")
    parser.add_argument('--memory',
                        type=int,
                        default=0,
                        help="0 -> Normal Replay buffer, 1 -> Priority Replay buffer(Not working with DQfD!) 2 -> single replay buffer")
    parser.add_argument('--humansize',
                        type=int,
                        default=50000,
                        help="size of human data")
    parser.add_argument('--actorsize',
                        type=int,
                        default=50000,
                        help="size of actor data")
    parser.add_argument('--rlstep',
                        type=int,
                        default=3000000,
                        help="how many train the model by merged data")
    parser.add_argument('--pretrainstep',
                        type=int,
                        default=750000,
                        help="how many pretrain the model by human data")
    parser.add_argument('--sampleschedule',
                        type=int,
                        default=5,
                        help="0 -> linear, 1 -> logarithm_fast(human exponential decay, decay step 5000) 2 -> logarithm(human exponential decay step 10000) 3 -> constant(0.75) 4 -> only actor, 5 -> mimic single replay management")
    parser.add_argument('--sampleschedulelength',
                        type=int,
                        default=1000000,
                        help="how long decay the sampling ratio")
    parser.add_argument('--rlstart',
                        type=int,
                        default=5000,
                        help=" Memory Buffer Size that Reinforcement Learning Start from")
    parser.add_argument('--pretrainstart',
                        type=int,
                        default=50000,
                        help="Memory Buffer Size that Pretrain  Start from")
    parser.add_argument('--lossmod',
                        type=int,
                        default=0,
                        help="Use Loss Model Modification Version 0(False) 1(True)")
    parser.add_argument('--lifereset',
                        type=int,
                        default=0,
                        help="Use Life Reset Mode 0(False) 1(True)")
    parser.add_argument('--rewardclip',
                        type=int,
                        default=0,
                        help="Use reward clipping 0(False) 1(True)")
    parser.add_argument('--optimizer',
                        type=int,
                        default=0,
                        help="Use optimizer adam 0 rmsprop 1")
    parser.add_argument('--lossclip',
                        type=int,
                        default=1,
                        help="Use lossclip 0(False) 1(True)")
    parser.add_argument('--usedfs',
                        type=int,
                        default=0,
                        help="Use usedfs 0(False) 1(True)")
    parser.add_argument('--dynamic_nq',
                        type=int,
                        default=1,
                        help="Use dynamic n q reward 0(False) 1(True)")

    args = parser.parse_args()
    testname = args.testname
    model = args.model
    memory = args.memory
    humansize = args.humansize
    actorsize = args.actorsize
    rlstep = args.rlstep + 1
    pretrainstep = args.pretrainstep + 1
    sampleschedule = args.sampleschedule
    sampleschedulelength = args.sampleschedulelength
    rlstart = args.rlstart
    pretrainstart = args.pretrainstart
    optimizer = args.optimizer
    dynamic_nq = args.dynamic_nq

    lossmod = args.lossmod
    lifereset = args.lifereset
    rewardclip = args.rewardclip
    lossclip = args.lossclip
    usedfs = args.usedfs

    USE_MODEL = model
    # if model == 1 : USE_DQfD = True
    # elif model == 0 : USE_DQfD = False

    USE_PRIORITY_REPLAY = False
    USE_SINGLE_REPLAY = False
    if memory == 1:
        USE_PRIORITY_REPLAY = True
    elif memory == 0:
        USE_PRIORITY_REPLAY = False
    elif memory == 2:
        USE_SINGLE_REPLAY = True

    HUMAN_MEMORY_SIZE = humansize
    ACTOR_MEMORY_SIZE = actorsize
    RL_STEP = rlstep
    PRETRAIN_STEP = pretrainstep
    RL_START = rlstart
    PRETRAIN_START = pretrainstart

    USE_LINEAR = False
    USE_LOGARITHM_FAST = False
    USE_LOGARITHM = False
    USE_CONSTANT = False
    USE_ONLY_ACTOR = False
    USE_MIMIC_SINGLE_REPLAY = False
    if sampleschedule == 0:
        USE_LINEAR = True
    elif sampleschedule == 1:
        USE_LOGARITHM_FAST = True
    elif sampleschedule == 2:
        USE_LOGARITHM = True
    elif sampleschedule == 3:
        USE_CONSTANT = True
    elif sampleschedule == 4:
        USE_ONLY_ACTOR = True
    elif sampleschedule == 5:
        USE_MIMIC_SINGLE_REPLAY = True
    DECAY_LENGTH = sampleschedulelength

    LOSS_MOD = False
    if (lossmod == 1): LOSS_MOD = True

    LIFE_RESET = False
    if (lifereset == 1): LIFE_RESET = True

    ACTOR_ITR = 1
    HUMAN_ITR = 1
    if (USE_ONLY_ACTOR == True):
        HUMAN_ITR = 0

    REWARD_CLIP = False
    if (rewardclip == 1):
        REWARD_CLIP = True
    LOSSCLIP = False
    if (lossclip == 1):
        LOSSCLIP = True
    OPTIMIZER = optimizer

    USE_DFS = False
    if (usedfs == 1):
        USE_DFS = True

    USE_DYNAMIC_N_Q = False
    if (dynamic_nq == 1):
        USE_DYNAMIC_N_Q = True

    session = tf.InteractiveSession()
    env = gym.make(Config.ENV_NAME)

    scores, e, replay_full_episode = [], 0, None
    gameName = Config.GAME_NAME
    gameID = Config.ENV_NAME
    dataSetAction = Config.ACTION_SET
    env = gym.make(gameID)
    gymAction = env.unwrapped.get_action_meanings()
    actionTranslator = actionTranslate(gymAction, dataSetAction)
    episodeList = os.listdir(Config().TRAJ_PATH + gameName + '/')  # dir is your directory path
    demo_size = 0

    if (USE_SINGLE_REPLAY):
        memory = SingleReplayBuffer_Custom(HUMAN_MEMORY_SIZE, ACTOR_MEMORY_SIZE, False, REWARD_CLIP, env.action_space.n,
                                           USE_DFS,
                                           USE_DYNAMIC_N_Q)
        human_memory = actor_memory = memory
    else:
        if (USE_PRIORITY_REPLAY):
            human_memory = PrioritizedReplayBuffer_Custom(HUMAN_MEMORY_SIZE, Config.ALPHA, True, REWARD_CLIP,
                                                          env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)
            actor_memory = PrioritizedReplayBuffer_Custom(ACTOR_MEMORY_SIZE, Config.ALPHA, False, REWARD_CLIP,
                                                          env.action_space.n, USE_DFS, USE_DYNAMIC_N_Q)
        else:
            human_memory = ReplayBuffer_Custom(HUMAN_MEMORY_SIZE, True, REWARD_CLIP, env.action_space.n, USE_DFS,
                                               USE_DYNAMIC_N_Q)
            actor_memory = ReplayBuffer_Custom(ACTOR_MEMORY_SIZE, False, REWARD_CLIP, env.action_space.n, USE_DFS,
                                               USE_DYNAMIC_N_Q)

    replay_manager = ReplayMemoryManager(human_memory, actor_memory, DECAY_LENGTH, USE_LINEAR, USE_LOGARITHM_FAST,
                                         USE_LOGARITHM, USE_CONSTANT, USE_ONLY_ACTOR, USE_MIMIC_SINGLE_REPLAY,
                                         USE_PRIORITY_REPLAY, USE_SINGLE_REPLAY)

    episodeList = get_goodepisode(episodeList, 0.2)

    # logfile initialize
    #######################################################################################################
    sample_log = openLog(Config.LEARNER_DATA_PATH + 'sampleexp/', '[' + str(testname) + ']',
                         ['step', 'value', 'actor_value', 'human_value', 'age', 'demo', 'qvalue'])
    replay_log = openLog(Config.LEARNER_DATA_PATH + 'replaymemory/', '[' + str(testname) + ']',
                         ['step', 'root_priority', 'root_ts', 'root_demo', 'alpha', 'beta'])
    delete_log = openLog(Config.ACTOR_DATA_PATH + 'deletedexp/', '[' + str(testname) + ']',
                         ['step', 'train_itr', 'value', 'age', 'demo'])
    episode_log = openLog(Config.ACTOR_DATA_PATH + 'episodescore/', '[' + str(testname) + ']',
                          ['episode', 'score', 'train_step'])
    epochscore_log = openLog(Config.ACTOR_DATA_PATH + 'epochscore/', '[' + str(testname) + ']',
                             ['train_step', 'score', 'average_ep', 'average_step'])
    #######################################################################################################

    screenpath = Config.SCREEN_PATH
    trajpath = Config.TRAJ_PATH

    agent = DQfD('learner', env, DQfDConfig(), session, replay_manager, USE_MODEL, USE_PRIORITY_REPLAY, PRETRAIN_STEP,
                 RL_STEP, PRETRAIN_START, RL_START, LOSS_MOD, OPTIMIZER, LOSSCLIP, USE_DFS)
    env = gym.make(Config.ENV_NAME)

    trainer = Trainer('learner', env, agent, episodeList, sample_log, replay_log, delete_log,
                      episode_log, epochscore_log, LIFE_RESET, ACTOR_ITR, HUMAN_ITR, REWARD_CLIP, USE_DFS)
    trainer.init_pretrain()
    trainer.pretrain()
    if (USE_ONLY_ACTOR == True):
        human_memory = None
        replay_manager.humanMemory = None
        gc.collect()
    print('init run')
    trainer.init_run()
    print('run')
    trainer.run()
'''