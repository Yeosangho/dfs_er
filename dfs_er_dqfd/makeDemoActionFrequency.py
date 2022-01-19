from Config import Config
import numpy as np
import gym
import os
import time
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

def get_goodepisode(episodeList, percent):
    episodeEnd = False
    i = f = None
    epsidoe_list_count = 0

    score_dtype = [('key', int), ('score', float), ('freq', object), ('avg', int)]
    ep_score_array = np.zeros(episodeList.__len__(), dtype=score_dtype)

    action_repetition = 0


    episode_action_counts = []

    for _ in range(episodeList.__len__()):
        episode = episodeList[epsidoe_list_count]
        episode = episode.split('.')[0]
        action_counts = []
        sum_action = 0
        div = 0
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
                    sum_action += action_repetition+1 -j
                    div += 1
                action_repetition = 0
            oldAction = action
            i += 1
            score += reward

        #episode_action_counts.append(action_counts)
        ep_score_array[epsidoe_list_count] = (int(episode), -score, action_counts, i)
        epsidoe_list_count += 1
    ep_score_array = np.sort(ep_score_array, order='score')
    #print()

    #print(ep_score_array)
    print(np.sum((ep_score_array['avg'][:int(episodeList.__len__() * percent)])))
    print(ep_score_array['score'][:int(episodeList.__len__() * percent)])
    print(ep_score_array[0])
    print(ep_score_array[0][2].__len__())
    return ep_score_array


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

if __name__ == '__main__':
    a = 1
    b = 2
    c = a/b
    a =0
    b= 0
    print(c)
    gameName = 'spaceinvaders'
    gameID = Config.ENV_NAME
    dataSetAction = Config.ACTION_SET
    env = gym.make(gameID)
    starttime = time.time()
    env.reset()
    for _ in range(20000):
        env.step(0)
    print(starttime - time.time())
    gymAction = env.unwrapped.get_action_meanings()
    actionTranslator = actionTranslate(gymAction, dataSetAction)
    episodeList = os.listdir(Config().TRAJ_PATH + gameName + '/')  # dir is your directory path
    demo_size = 0


    episodeList = get_goodepisode(episodeList, 0.06)
    print(episodeList[0])