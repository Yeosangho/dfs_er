
import matplotlib.pyplot as plt
import math
import gym
import random
import tensorflow as tf
from collections import deque
import numpy as np
def print_values(a, **kwargs):
    print(a)
    print(kwargs.get('my_name'))
    for key, value in kwargs.items():
        print("The value of {} is {}".format(key, value))


if __name__ == '__main__':
    print(np.log(100))
    print(np.log(10))
    print(np.log(25))
    a = np.asarray([9.9867588e-01, 2.4616835e-04, 2.7712630e-04, 2.4412177e-04, 3.1385338e-04,
 2.4261871e-04])
    v_hat_s =   np.sum(
        1e-20 * np.log(1e-20 + 1e-7))
    a = np.zeros((1), dtype='float16')
    a[0] = 1e-20
    print(type(a[0]))
    print(a[0])

    b = np.asarray([2,3,4])
    v = tf.log(1e-7)  # initialize a variable as nan  ​
    a = tf.add(v, v)
    with tf.Session() as sess:
        a = sess.run(a)
        print(a)
    print(a*b)
    count = 0
    for i in range(10):
        for i in range(4):
            count += 1
            print(count)
    print(-1%5)
    t_q_human = deque(maxlen=4)
    t_q_human.append([1,False])
    t_q_human.append([2,False])
    t_q_human.append([3,True])
    t_q_human.append([4,True])
    t_q_human.append([5, True])
    t_list = list(t_q_human)
    t_q_human.append([5, True])
    print( list(t_q_human))
    print(sum(t[0] for i, t in enumerate(t_list[t_q_human.__len__()-4:t_q_human.__len__()])))
    print( list(t_q_human))

    print(a)
    print(-math.log(1 + 100))
    print(32 - int((32 **((380000) / 500000))))
    print(random.randint(0, 1))
    i = 1
    for j in range(1000):
        i =i * 0.999
    print(i)


    x = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    minibatch = 32
    j = 0
    for i in range(1000000):
        x.append(i)
        y.append(int(32*i/1000000))
        #y1.append(int(32 - 32**( (1000000 - i) / 1000000)))
        if(i < 1000000):
            y1.append(int( 32**( ( i) / 1000000)))
        else:
            y1.append(32)
        #y2.append(int(32 * math.log(i+1 , 1000000)))
        if(i < 1000000):
            decayed_batch = 32 * 0.96 ** (i / 10000)
            y2.append(int(32-decayed_batch))
        else:
            y2.append(32)

        y4.append(32 *  (j/(12500+j)))
        decayed_batch = 32 * 0.96**(i/5000)
        y3.append(int(32 - decayed_batch))
        if(j<50000):
            j = j + 1

    yl, = plt.plot(x, y, label='dual_linear')
    yl2, = plt.plot(x, y2, label='dual_exp_slow', color='yellow')
    yl3, = plt.plot(x, y3, label='dual_exp_fast', color='green')
    yl4, = plt.plot(x, y4, label='single', color='orange')
    plt.legend(handles=[yl, yl2, yl3, yl4])

    plt.xlabel('step')
    plt.ylabel('minibatch')
    plt.title('Actor Batches in Each Decay Scheduling\n')
    plt.show()