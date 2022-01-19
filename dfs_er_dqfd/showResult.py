from Config import *
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import csv

def read_lines():
    #with open(Config.ACTOR_DATA_PATH +'epochscore/'+ '2019-01-22:00:47:33[revenge_100MIL_0.25sample_nopretrain_ofs].csv', 'rU') as data:
    with open(
                                Config.ACTOR_DATA_PATH + 'episodescore/' + '2019-01-27:03:45:19[revenge_100MIL_dqfd_no_ofs_0.5Mactorbuffer_per].csv',
                                'rU') as data:
    #with open(Config.LEARNER_DATA_PATH + 'sampleexp/' + '2018-07-23:18:20:31[\'dqfd_schedule0\'].csv', 'rU') as data:
        reader = csv.reader(data)
        skiprow = 1
        count =0
        for row in reader:
            count = count +1
            if(skiprow < count) :
                yield [ float(i) for i in row ]

time = []
score = []
step = 10
read_count = 0
avg_time = 0
avg_score = 0
for i in read_lines():
    avg_time += i[0]
    avg_score += i[1]
    if(read_count % 1 == 0):
        time.append(avg_time/1 )
        score.append(avg_score/1)
    avg_time = 0
    avg_score = 0

    read_count = read_count + 1

plt.plot(time,score)
plt.xlabel('Time')
plt.ylabel('Load Value')
plt.title('Logged Load\n')
plt.show()