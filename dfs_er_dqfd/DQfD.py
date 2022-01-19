# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
import functools
import ReplayMemoryManager
from ops import linear, conv2d, clipped_error
from functools import reduce
import time
def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


def updateNet( from_name, to_name):
    to_vars = tf.get_collection(to_name)
    from_vars = tf.get_collection(from_name)
    print("####" + str(to_vars))
    print("###"+ str(from_vars))
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def loss1():
    return 0.0
def loss2():
    return 0.8

class DQfD:
    def __init__(self, name, env, config, sess, memory_manager, USE_MODEL, USE_PRIORITY_REPLAY, PRETRAIN_STEP, RL_STEP, PRETRAIN_START, RL_START, LOSS_MOD, OPTIMIZER, LOSSCLIP, USE_DFS):
        self.USE_MODEL = USE_MODEL
        if(self.USE_MODEL == 2):
            self.alpha = config.NAC_ALPHA
            self.eps = 1e-15
            self.max = 1000.
        self.USE_PRIORITY_REPLAY = USE_PRIORITY_REPLAY
        self.pretrain_step = PRETRAIN_STEP
        self.rl_step = RL_STEP
        self.RL_START = RL_START
        self.PRETRAIN_START = PRETRAIN_START
        self.LOSS_MOD = LOSS_MOD
        self.optimizer = OPTIMIZER
        self.lossclip = LOSSCLIP
        self.USE_DFS = USE_DFS
        self.sess = sess
        self.config = config
        self.name = name
        self.history_length = config.HISTORY_LENGTH
        self.batch_size = config.BATCH_SIZE
        # replay_memory stores both demo data and generated data, while demo_memory only store demo data
        #Demo Data Part
        #self.replay_memory = Memory(capacity=self.config.replay_buffer_size, permanent_data=len(demo_transitions))
        #self.demo_memory = Memory(capacity=self.config.demo_buffer_size, permanent_data=self.config.demo_buffer_size)
        self.memory_manager = memory_manager
        #self.demo_memory = Memory(capacity=self.config.demo_buffer_size)

        #No Data Data
        #self.add_demo_to_memory(demo_transitions=demo_transitions)  # add demo data to both demo_memory & replay_memory

        self.time_step = 0
        self.model_score = 0

        self.epsilon = self.config.INITIAL_EPSILON
        self.actions = env.action_space.n
        if(self.USE_DFS):
            self.action_dim = env.action_space.n * 2
        else:
            self.action_dim = env.action_space.n
        self.action_freq = self.config.ACTION_FREQ_RANGE
        self.action_batch = tf.placeholder("int32", [None])
        self.reward_batch = tf.placeholder("float", [None])
        self.freq_batch = tf.placeholder("int32", [None])
        self.tf_model_score = tf.placeholder("float", [None])
        self.y_input = tf.placeholder("float", [None])
        self.ISWeights = tf.placeholder("float", [None, 1])
        self.n_step_y_input = tf.placeholder("float", [None])  # for n-step reward
        self.isdemo = tf.placeholder("float", [None])
        self.eval_input = tf.placeholder("float", [None, self.history_length, 84,84])
        self.select_input = tf.placeholder("float", [None, self.history_length, 84,84])

        self.y_freq_input = tf.placeholder("float", [None, self.action_freq])
        self.n_step_y_freq_input = tf.placeholder("float", [None, self.action_freq])

        self.learning_rate_step = tf.placeholder('int32', [None])


        self.v_hat_s = tf.placeholder("float32", [None])
        self.policy_q_select = tf.placeholder("float32", [None, self.action_dim])
        self.v_q_s_select = tf.placeholder("float32", [None])
        self.q_hat_s_a = tf.placeholder("float32", [None, self.action_dim])
        self.q_hat = tf.placeholder("float32", [None, self.action_dim])


        self.gradient = tf.placeholder("float32", [None])

        #action_dim
        self.expert_loss = []
        for i in range(self.action_dim) :
            loss = [0.8] * self.action_dim
            loss[i] -= 0.8
            self.expert_loss.append(loss)
        self.expert_loss = tf.constant(self.expert_loss, dtype="float")


        self.Q_eval
        self.Q_select

        self.loss
        self.optimize

        self.update_target_net
        self.abs_errors


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if(self.config.START_STEP == 0):
            self.save_model()

        self.dueling = True
        self.demo_num = 0
        self.sum_abs_error = 0
        self.sum_abs_error_freq = 0
        self.sum_actor_batch = 0
        self.sum_actor_abs_error = 0
        self.sum_human_batch = 0
        self.sum_human_abs_error = 0
        self.sum_age = 0
        self.qvalue = [0.0] * self.action_dim
        self.oldsum = 0
    def isHumanFull(self):
        return self.memory_manager.isHumanFull()

    def isActorFull(self):
        return self.memory_manager.isActorFull()

    def initPretrain(self):
        print(self.memory_manager.getHumanMemoryLength())
        if (self.memory_manager.getHumanMemoryLength() < self.PRETRAIN_START):
            return True
        else:
            return False

    def initRL(self):
        if (self.memory_manager.getActorMemoryLength() < self.RL_START):
            return True
        else:
            return False

    def isPretrain(self):
        if(self.time_step > self.pretrain_step):
            return False
        if(self.memory_manager.getHumanMemoryLength() >= self.PRETRAIN_START):
            return True
        else:
            return False
    def isRL(self):
        if(self.time_step > (self.rl_step + self.pretrain_step)):
            return False
        if(self.memory_manager.getActorMemoryLength() >= self.RL_START):
            return True
        else:
            return False

    def getTimeStep(self):
        return self.time_step

    def add_demo_to_memory(self, demo_transitions):
        # add demo data to both demo_memory & replay_memory
        for t in demo_transitions:
            self.demo_memory.store(np.array(t, dtype=object))
            self.replay_memory.store(np.array(t, dtype=object))
            assert len(t) == 10

    def build_layers(self, states, cnames, initializer, activation_fn, network_type="cnn", reg=None):
        self.w = {}
        self.t_w = {}
        self.dueling = True
        a_d = self.action_dim

        self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(states,
                                                     32, [8, 8], [4, 4], cnames,initializer, activation_fn,
                                                        name='l1')
        self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                                                     64, [4, 4], [2, 2],  cnames, initializer, activation_fn,
                                                        name='l2')
        self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                                                     64, [3, 3], [1, 1], cnames, initializer, activation_fn,
                                                        name='l3')

        shape = self.l3.get_shape().as_list()
        self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        if self.dueling:
            self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                linear(self.l3_flat, 1024, cnames, activation_fn=activation_fn, name='value_hid')

            self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                linear(self.l3_flat, 1024, cnames, activation_fn=activation_fn, name='adv_hid')

            self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                linear(self.value_hid, 1, cnames, name='value_out')

            self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                linear(self.adv_hid, self.action_dim, cnames, name='adv_out')

            # Average Dueling
            self.q = self.value +  (self.advantage -  tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        else:
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 1024, cnames, activation_fn=activation_fn, name='l4')

            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_dim, cnames, name='q')

        return self.q

    @lazy_property
    def Q_select(self):
        with tf.variable_scope(self.name+'_select_net') as scope:
            c_names = [self.name+'_select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            initializer = tf.truncated_normal_initializer(0, 0.02)
            activation_fn = tf.nn.relu
            reg = tf.contrib.layers.l2_regularizer(scale=0.2)  # Note: only parameters in select-net need L2
            return self.build_layers(self.select_input, c_names, initializer, activation_fn, reg)

    @lazy_property
    def Q_eval(self):
        with tf.variable_scope(self.name+'_eval_net') as scope:
            c_names = [self.name+'_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            initializer = tf.truncated_normal_initializer(0, 0.02)
            activation_fn = tf.nn.relu
            return self.build_layers(self.eval_input, c_names, initializer, activation_fn)



    @lazy_property
    def Q_eval_action(self):
        self.Q_eval_action = tf.argmax(self.Q_eval, dimension=1)

    @lazy_property
    def Q_select_action(self):
        self.Q_select_action = tf.argmax(self.Q_select, dimension=1)





    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self, Q_select):
        jeq = 0.0
        for i in range(self.config.BATCH_SIZE):
            ae = self.action_batch[i]
            max_value = float("-inf")
            for a in range(self.action_dim):
                max_value = tf.maximum(Q_select[i][a] + self.loss_l(ae, a), max_value)
            jeq += self.isdemo[i] * (max_value - Q_select[i][ae])
        return jeq


    @lazy_property
    def loss(self):

        # prepare row indices
        row_indices = tf.range(tf.shape(self.action_batch)[0])

        # zip row indices with column indices
        full_indices = tf.stack([row_indices, self.action_batch], axis=1)

        # retrieve values by indices
        Q_select_action_batch = tf.gather_nd(self.Q_select, full_indices)

        dq_delta = self.y_input - Q_select_action_batch
        n_dq_delta = self.n_step_y_input - Q_select_action_batch
        l_dq = tf.reduce_mean(clipped_error(dq_delta))
        l_n_dq = tf.reduce_mean(clipped_error(n_dq_delta))
        l_jeq = 0.0

        e_loss = self.expert_loss
        expert_loss = tf.add(self.Q_select, tf.gather(e_loss, self.action_batch))
        max_value = tf.reduce_max(expert_loss, axis=1)


        l_jeq_mb = self.isdemo * (max_value -Q_select_action_batch)
        l_jeq = tf.reduce_mean(l_jeq_mb)

        
        l_l2 = tf.reduce_sum(
            [tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        l_action = 0.0
        if (self.USE_PRIORITY_REPLAY == True):
            loss =  self.ISWeights * tf.reduce_sum([l * λ for l, λ in zip([l_dq, l_n_dq, l_jeq, l_l2], self.config.LAMBDA)])
        else :
            loss =  tf.reduce_sum([l * λ for l, λ in zip([l_dq, l_n_dq, l_jeq, l_l2], self.config.LAMBDA)])

        return loss, tf.abs(dq_delta+n_dq_delta)



    @lazy_property
    def abs_errors(self):
        # prepare row indices
        row_indices = tf.range(tf.shape(self.action_batch)[0])

        # zip row indices with column indices
        full_indices = tf.stack([row_indices, self.action_batch], axis=1)

        # retrieve values by indices
        Q_select_action_batch = tf.gather_nd(self.Q_select, full_indices)
        return tf.abs(self.y_input - Q_select_action_batch)  # only use 1-step R to compute abs_errors



    @lazy_property
    def optimize(self):

        if(self.optimizer == 0):
            optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        elif(self.optimizer == 1):
            learning_rate_op = tf.maximum(self.config.LEARNING_RATE_FINAL,
                                               tf.train.polynomial_decay(
                                                   self.config.LEARNING_RATE,
                                                   self.learning_rate_step[0],
                                                   self.config.LEARNING_RATE_DECAY_STEP))
            optimizer = tf.train.RMSPropOptimizer(learning_rate_op, momentum=0.95, epsilon=0.01)


        return optimizer.minimize(self.loss[0]), self.loss[1]  # only parameters in select-net is optimized here

    @lazy_property
    def apply_gradient(self):
        if(self.optimizer == 0):
            optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        elif(self.optimizer == 1):
            learning_rate_op = tf.maximum(self.config.LEARNING_RATE_FINAL,
                                               tf.train.polynomial_decay(
                                                   self.config.LEARNING_RATE,
                                                   self.learning_rate_step[0],
                                                   self.config.LEARNING_RATE_DECAY_STEP))
        if (self.USE_MODEL == 2):
            train_op = optimizer.apply_gradients(self.gradient)
            return train_op, [self.loss, tf.constant(0)]

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection(self.name +'_select_net_params')
        eval_params = tf.get_collection(self.name +'_eval_net_params')
        print("update_eval_ops" + str(eval_params))
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def save_model(self):
        if(self.name == 'learner'):
            print("Model saved in : {}".format(self.saver.save(self.sess, self.config.MODEL_PATH,  global_step = self.time_step+self.config.START_STEP)))

    def restore_model(self):
        if(self.name ==  'learner'):
            self.saver.restore(self.sess, '/home/soboru963/PycharmProjects/0621_PriorityBasedUpdate/DQfD_model-389103')
            print("Model restored.")

    def perceive(self, transition, score):

        self.memory_manager.add(transition)



    def train_Q_network_with_update_freq(self):
        if(self.time_step % self.config.UPDATE_FREQ == 0):
            self.train_Q_network();
        self.time_step += 1

    def train_Q_network(self):
        """
        :param pre_train: True means should sample from demo_buffer instead of replay_buffer
        :param update: True means the action "update_target_net" executes outside, and can be ignored in the function
        """



        ##For the Pretrain
        #actual_memory = self.demo_memory if pre_train else self.replay_memory
        if(self.USE_PRIORITY_REPLAY == True):
            tree_idxes, minibatch, ISWeights = self.memory_manager.sample(self.time_step, self.pretrain_step)
        else:
            minibatch = self.memory_manager.sample(self.time_step, self.pretrain_step)
        minibatch_trans = minibatch.transpose()

        state_batch = np.array(list(minibatch_trans[0]), dtype=np.float32) / 255.0
        action_batch = np.array(list(minibatch_trans[1]), dtype=np.int32)
        reward_batch =  np.array(list(minibatch_trans[2]), dtype=np.float32)
        next_state_batch = np.array(list(minibatch_trans[3]), dtype=np.float32) / 255.0
        done_batch = np.array(list(minibatch_trans[4]), dtype=np.int32)
        demo_data = np.array(list(minibatch_trans[5]), dtype=np.float32)
        n_step_reward_batch = np.array(list(minibatch_trans[6]), dtype=np.float32)
        n_step_state_batch = np.array(list(minibatch_trans[7]), dtype=np.float32) / 255.0
        n_step_done_batch = np.array(list(minibatch_trans[8]), dtype=np.int32)
        actual_n = np.array(list(minibatch_trans[9]), dtype=np.float32)


        whole_state_bat = []
        whole_state_bat = np.concatenate((next_state_batch, n_step_state_batch), axis=0)
        whole_eval_bat =  np.concatenate((next_state_batch, n_step_state_batch), axis=0)
        whole_Q_select, whole_Q_eval = self.sess.run([self.Q_select, self.Q_eval], feed_dict={self.select_input : whole_state_bat, self.eval_input : whole_eval_bat})

        Q_select, n_step_Q_select = whole_Q_select[0:self.config.BATCH_SIZE], whole_Q_select[self.config.BATCH_SIZE: self.config.BATCH_SIZE *2]
        Q_eval, n_step_Q_eval  = whole_Q_eval[0:self.config.BATCH_SIZE], whole_Q_eval[self.config.BATCH_SIZE: self.config.BATCH_SIZE *2]
        

        action = np.argmax(Q_select, axis=1)
        Q_eval_max_action = Q_eval[np.arange(len(Q_eval)), action]


        y_batch = reward_batch + (1-done_batch) * self.config.GAMMA * Q_eval_max_action

        action = np.argmax(n_step_Q_select, axis=1)
        n_step_Q_eval_max_action = n_step_Q_eval[np.arange(len(n_step_Q_eval)), action]
        q_n_step = (1-n_step_done_batch) * self.config.GAMMA ** np.array(actual_n) * n_step_Q_eval_max_action
        n_step_y_batch =  n_step_reward_batch + q_n_step

        if (self.USE_PRIORITY_REPLAY == True):
            a, abs_errors = self.sess.run(self.optimize,
                                          feed_dict={self.y_input: y_batch,
                                                     self.n_step_y_input: n_step_y_batch,
                                                     self.select_input: state_batch,
                                                     self.action_batch: action_batch,
                                                     self.reward_batch:reward_batch,
                                                     self.learning_rate_step: [self.time_step],
                                                     self.ISWeights : ISWeights,
                                                   self.isdemo: demo_data})
        else :
            a, abs_errors = self.sess.run(self.optimize,
                                          feed_dict={self.y_input: y_batch,
                                                     self.n_step_y_input: n_step_y_batch,
                                                     self.select_input: state_batch,
                                                     self.action_batch: action_batch,
                                                     self.reward_batch:reward_batch,
                                                     self.learning_rate_step: [self.time_step],
                                                   self.isdemo: demo_data})
        if (self.USE_PRIORITY_REPLAY == True):
            self.memory_manager.update_priorities(tree_idxes, abs_errors, demo_data)  # update priorities for data in memory
        #print(abs_errors)

    def egreedy_action(self, history):
        if (self.isRL()):
            self.epsilon = max(self.config.FINAL_EPSILON, self.epsilon * self.config.EPSILIN_DECAY)
        #sum =0
        #for i in a[5]:
        #    sum = sum + i
        #print(sum)
        history = history/255.0
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        #print('no_random')
        action = np.argmax(self.sess.run(self.Q_select, feed_dict={self.select_input: [history]})[0])
        #print(action)
        return action


    def setLocalNet(self):
        self.sess.run(self.update_local_ops)
    def getSelectNet(self):
        return self.sess.run(tf.get_collection(self.name+'_select_net_params'))








