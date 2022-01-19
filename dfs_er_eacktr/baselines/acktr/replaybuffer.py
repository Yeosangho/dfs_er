from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random
import math
def sign(x): return 1 if x >= 0 else -1
class ReplayBuffer_Custom(ReplayBuffer):
    def __init__(self, size, IS_HUMAN, REWARD_CLIP, ACTION_NUM, USE_DFS, DYNAMIC_N_Q_STEP):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(ReplayBuffer_Custom, self).__init__(size)
        self.actions = [0] * size
        self.rewards = [0.0] * size
        self.states = np.zeros((size, Config.EXP_LENGTH ,84, 84), dtype=np.uint8)
        self.dones =  [False] * size
        self.ishumans = [0.0] * size
        self.timesteps = [0] * size
        if(IS_HUMAN):
            self.r1_rewards = [0.0] * size
            self.step_in_ep = [0] * size
            if(USE_DFS):
                self.r2_rewards =[0.0] *size
        if(not IS_HUMAN):
            self.life_loss = [False] * size
        self.current_storage_len = 0
        self.IS_HUMAN = IS_HUMAN
        self.REWARD_CLIP = REWARD_CLIP
        self.ACTION_NUM = ACTION_NUM
        self.USE_DFS = USE_DFS
        self.SKIP_R1 = Config.SKIP_R1
        self.SKIP_R2 = Config.ACTION_FREQ_RANGE
        self.DYNAMIC_N_Q_STEP = DYNAMIC_N_Q_STEP
        if(DYNAMIC_N_Q_STEP):
            self.DYNAMIC_N = int(self.SKIP_R2 / self.SKIP_R1)
            self.MAX_N_STEP = self.SKIP_R1 * 10

        self.n_step = Config.N_STEP
        self.count = 0
    def __len__(self):
        return self.current_storage_len

    def add(self, data):
        if self._next_idx >= self.current_storage_len:
            self.current_storage_len += 1
        self.actions[self._next_idx] = data[0]
        self.rewards[self._next_idx] = data[1]
        self.states[self._next_idx] = data[2]
        self.dones[self._next_idx] = data[3]
        self.ishumans[self._next_idx] = data[4]
        if(self.IS_HUMAN):
            self.r1_rewards[self._next_idx] = data[5]
            self.step_in_ep[self._next_idx] = data[6]
            if(self.USE_DFS):
                self.r2_rewards[self._next_idx] = data[7]
        if(not self.IS_HUMAN):
            self.life_loss[self._next_idx] = data[5]

        #self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        b_memory = np.empty((idxes.__len__(), 10), dtype=object)
        #experience = []
        batch_idx = 0
        for i in idxes:
            data = self.get_data(i)
            b_memory[batch_idx] = data
            batch_idx += 1
        return b_memory

    def get_data(self, idx):
        if(self.IS_HUMAN):
            if(self.SKIP_R1 == 1):
                self.get_human_data_noskip(idx)
            if(self.USE_DFS):
                return self.get_human_data_dfs(idx)
            else:
                return self.get_human_data(idx)
        else:
            return self.get_actor_data(idx)
    def get_actor_data(self, idx):
        action = self.actions[idx % self._maxsize]
        reward = self.rewards[idx % self._maxsize]
        prestates = self.get_actor_state(idx-1)
        poststates = self.get_actor_state(idx)

        done = self.dones[idx % self._maxsize] or self.life_loss[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]

        n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_data(idx)
        return [prestates, action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done,
                actual_n]
    def get_actor_state(self, idx):
        return np.asarray([self.states[(idx-3) % self._maxsize][0],
                self.states[(idx-2) % self._maxsize][0],
                self.states[(idx-1) % self._maxsize][0],
                self.states[idx % self._maxsize][0]])
    def get_nstep_data(self, idx):
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        for i in range(self.n_step):
            if(not (self.dones[(idx +i) % self._maxsize] or self.life_loss[(idx +i) % self._maxsize])):
                n_step_reward +=  self.rewards[(idx +i) % self._maxsize] * (Config.GAMMA ** i)
            else:
                n_step_done = True
                n_step_reward -= self.rewards[(idx + (i - 1)) % self._maxsize] * ( Config.GAMMA ** (i - 1))
                actual_n = i -1
                break
        if(n_step_done == False):
            if (not (self.dones[(idx + self.n_step) % self._maxsize]  or self.life_loss[(idx +self.n_step) % self._maxsize])):
                actual_n = self.n_step
            else:
                n_step_done = True
                n_step_reward -= self.rewards[(idx + (self.n_step - 1)) % self._maxsize] * (Config.GAMMA ** (self.n_step - 1))
                actual_n = self.n_step - 1
        n_step_state = self.get_actor_state(idx + actual_n)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_human_data_noskip(self, idx):
        action = self.actions[idx % self._maxsize]
        reward = self.rewards[idx % self._maxsize]
        prestates = self.get_actor_state(idx-1)
        poststates = self.get_actor_state(idx)

        done = self.dones[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]

        n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_human_data_noskip(idx)
        return [prestates, action, reward, poststates, done, ishuman,  n_step_reward, n_step_state, n_step_done,
                actual_n]



    def get_human_data(self, idx):
        action = self.actions[idx % self._maxsize]
        reward = self.r1_rewards[(idx) % self._maxsize]
        prestates = self.get_human_states((idx -self.SKIP_R1))
        poststates = self.get_human_states(idx)
        
        done = self.dones[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]

        n_step_reward,  n_step_done, actual_n, n_step_state = self.get_nstep_human_data(idx)
        
        return [prestates,action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done, actual_n]

    def get_nstep_human_data(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        for i in range(self.n_step):
            current_ep_count =  self.step_in_ep[(idx + (i*self.SKIP_R1)) % self._maxsize]
            if(ep_count <= current_ep_count):
                n_step_reward +=  self.r1_rewards[(idx +i*self.SKIP_R1) % self._maxsize] * (Config.GAMMA ** i)
            else:
                n_step_done = True
                n_step_reward -= self.r1_rewards[(idx + (i - 1)*self.SKIP_R1) % self._maxsize] * ( Config.GAMMA ** (i - 1))
                actual_n = i -1
                break
        if(n_step_done == False):
            current_ep_count =  self.step_in_ep[(idx + ((self.n_step)*self.SKIP_R1)) % self._maxsize]
            if(ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                n_step_reward -= self.r1_rewards[(idx + (self.n_step - 1)*self.SKIP_R1) % self._maxsize] * (Config.GAMMA ** (self.n_step - 1))
                actual_n = self.n_step - 1
        n_step_state = self.get_human_states((idx + actual_n * self.SKIP_R1) % self._maxsize)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_nstep_human_data_noskip(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False

        actual_n = 0
        for i in range(self.n_step):
            current_ep_count =  self.step_in_ep[(idx + (i*self.SKIP_R1)) % self._maxsize]
            if(ep_count <= current_ep_count):
                reward = self.rewards[(idx  + (i*self.SKIP_R1)) % self._maxsize] * (Config.GAMMA ** i)
                n_step_reward += reward
            else :
                n_step_done = True
                reward = self.rewards[(idx  + ((i-1)*self.SKIP_R1)) % self._maxsize] * (Config.GAMMA ** (i-1))
                n_step_reward -=  reward
                actual_n = i-1
                break
        if(n_step_done == False):
            current_ep_count =  self.step_in_ep[(idx + ((self.n_step)*self.SKIP_R1)) % self._maxsize]
            if(ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                reward = self.rewards[(idx  + ((self.n_step-1)*self.SKIP_R1)) % self._maxsize] * (Config.GAMMA ** (self.n_step-1))
                n_step_reward -=  reward
                actual_n = self.n_step - 1
        n_step_state = self.get_actor_state((idx + actual_n*self.SKIP_R1) % self._maxsize)
        return n_step_reward, n_step_done, actual_n, n_step_state


    def get_human_data_dfs(self, idx):
        action = self.actions[idx % self._maxsize]

        if(self.actions[(idx - self.SKIP_R2)% self._maxsize] >= self.ACTION_NUM):
            prestates = self.get_human_states_dfs(idx - self.SKIP_R2)
            reward = self.r2_rewards[(idx) % self._maxsize]
        else:
            prestates = self.get_human_states_dfs((idx - self.SKIP_R1))
            reward = self.r1_rewards[(idx) % self._maxsize]
        poststates = self.get_human_states(idx)

        done = self.dones[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]
        if(self.DYNAMIC_N_Q_STEP):
            n_step_reward, n_step_done, actual_n, n_step_state = self.get_dynamic_nstep_human_data_dfs(idx)
        else:
            n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_human_data_dfs(idx)

        return [prestates, action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done,
                actual_n]

    def get_human_states_dfs(self, idx):
        states = np.zeros((Config.HISTORY_LENGTH ,84, 84), dtype=np.uint8)
        index = idx
        state = self.states[(index) % self._maxsize][0]
        states[Config.HISTORY_LENGTH -1] = state
        for i in range(Config.HISTORY_LENGTH-1):
            if (self.actions[(index - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
                state = self.states[(index - self.SKIP_R2) % self._maxsize][0]
                states[Config.HISTORY_LENGTH-2-i] = state
                index = index - self.SKIP_R2
            else:
                state = self.states[(index - self.SKIP_R1) % self._maxsize][0]
                states[Config.HISTORY_LENGTH-2-i] = state
                index = index - self.SKIP_R1
        return states

    def get_dynamic_nstep_human_data_dfs(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        indexes = []
        sum_dynamic_n = 0
        if (self.actions[(idx - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
            index = idx - self.SKIP_R2
        else:
            index = idx - self.SKIP_R1
        indexes.append(index)
        index = idx
        indexes.append(index)
        action = self.actions[index % self._maxsize]


        for i in range(self.n_step):

            current_ep_count =  self.step_in_ep[(indexes[i+1]) % self._maxsize]
            if(ep_count <= current_ep_count):
                if(abs(indexes[i+1]-indexes[i]) == self.SKIP_R2):
                    n_step_reward += self.r2_rewards[(indexes[i+1]) % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n))
                else:
                    n_step_reward +=  self.r1_rewards[(indexes[i+1]) % self._maxsize] * (Config.GAMMA ** sum_dynamic_n)
            else :
                n_step_done = True
                if (abs(indexes[i - 1] - indexes[i]) == self.SKIP_R2):
                    sum_dynamic_n = sum_dynamic_n - self.DYNAMIC_N
                    n_step_reward -=  self.r2_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n))
                else:
                    sum_dynamic_n = sum_dynamic_n - 1
                    n_step_reward -= self.r1_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n ))
                actual_n = i-1
                break
            if(action >= self.ACTION_NUM):
                index = index + self.SKIP_R2
                indexes.append(index)
                sum_dynamic_n = sum_dynamic_n + self.DYNAMIC_N

            else:
                index = index + self.SKIP_R1
                indexes.append(index)
                sum_dynamic_n = sum_dynamic_n + 1
            action = self.actions[index % self._maxsize]

        if(n_step_done == False):
            current_ep_count =  self.step_in_ep[indexes[self.n_step+1] % self._maxsize]

            if(ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                if (abs(indexes[self.n_step - 1] - indexes[self.n_step]) == self.SKIP_R2):
                    sum_dynamic_n -= self.DYNAMIC_N
                    n_step_reward -=  self.r2_rewards[indexes[self.n_step] % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n))
                else:
                    sum_dynamic_n -= 1
                    n_step_reward -= self.r1_rewards[indexes[self.n_step] % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n))
                actual_n = self.n_step - 1
        n_step_state = self.get_human_states_dfs(indexes[actual_n+1] % self._maxsize)
        return n_step_reward, n_step_done, sum_dynamic_n, n_step_state
    def get_nstep_human_data_dfs(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        indexes = []

        if (self.actions[(idx - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
            index = idx - self.SKIP_R2
        else:
            index = idx - self.SKIP_R1
        indexes.append(index)
        index = idx
        indexes.append(index)
        action = self.actions[index % self._maxsize]


        for i in range(self.n_step):

            current_ep_count =  self.step_in_ep[(indexes[i+1]) % self._maxsize]
            if(ep_count <= current_ep_count):
                if(abs(indexes[i+1]-indexes[i]) == self.SKIP_R2):
                    n_step_reward += self.r2_rewards[(indexes[i+1]) % self._maxsize] * (Config.GAMMA ** (i))
                else:
                    n_step_reward +=  self.r1_rewards[(indexes[i+1]) % self._maxsize] * (Config.GAMMA ** i)
            else :
                n_step_done = True
                if (abs(indexes[i - 1] - indexes[i]) == self.SKIP_R2):
                    n_step_reward -=  self.r2_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (i-1))
                else:
                    n_step_reward -= self.r1_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (i - 1))
                actual_n = i-1
                break
            if(action >= self.ACTION_NUM):
                index = index + self.SKIP_R2
                indexes.append(index)

            else:
                index = index + self.SKIP_R1
                indexes.append(index)
            action = self.actions[index % self._maxsize]

        if(n_step_done == False):
            current_ep_count =  self.step_in_ep[indexes[self.n_step+1] % self._maxsize]

            if(ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                if (abs(indexes[self.n_step - 1] - indexes[self.n_step]) == self.SKIP_R2):
                    n_step_reward -=  self.r2_rewards[indexes[self.n_step] % self._maxsize] * (Config.GAMMA ** (self.n_step-1))
                else:
                    n_step_reward -= self.r1_rewards[indexes[self.n_step] % self._maxsize] * (Config.GAMMA ** (self.n_step - 1))
                actual_n = self.n_step - 1
        n_step_state = self.get_human_states_dfs(indexes[actual_n+1] % self._maxsize)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_human_states(self, idx):
        return np.asarray([self.states[(idx-self.SKIP_R1 *3) % self._maxsize][0],
                self.states[(idx-self.SKIP_R1 *2) % self._maxsize][0],
                self.states[(idx-self.SKIP_R1 *1) % self._maxsize][0],
                self.states[idx % self._maxsize][0]])

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, self.current_storage_len - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def getStorageLength(self):
        return self.current_storage_len


class PrioritizedReplayBuffer_Custom(PrioritizedReplayBuffer):
    def __init__(self, size, alpha, IS_HUMAN, REWARD_CLIP, ACTION_NUM, USE_DFS, DYNAMIC_N_Q_STEP):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer_Custom, self).__init__(size, alpha)
        self.human_eps = Config.HUMAN_EPSILON
        self.actor_eps = Config.ACTOR_EPSILON
        self.actions = [0] * size
        self.rewards = [0.0] * size
        self.states = np.zeros((size, Config.EXP_LENGTH ,84, 84), dtype=np.uint8)
        self.dones =  [False] * size
        self.ishumans = [0.0] * size
        self.timesteps = [0] * size
        if(IS_HUMAN):
            self.r1_rewards = [0.0] * size
            self.step_in_ep = [0] * size
            if(USE_DFS):
                self.r2_rewards =[0.0] *size
        if(not IS_HUMAN):
            self.life_loss = [False] * size
        self.current_storage_len = 0
        self.IS_HUMAN = IS_HUMAN
        self.REWARD_CLIP = REWARD_CLIP
        self.ACTION_NUM = ACTION_NUM
        self.USE_DFS = USE_DFS
        self.SKIP_R1 = Config.SKIP_R1
        self.SKIP_R2 = Config.ACTION_FREQ_RANGE
        self.DYNAMIC_N_Q_STEP = DYNAMIC_N_Q_STEP
        if(DYNAMIC_N_Q_STEP):
            self.DYNAMIC_N = int(self.SKIP_R2 / self.SKIP_R1)
            self.MAX_N_STEP = self.SKIP_R1 * 10

        self.n_step = Config.N_STEP
        self.count = 0

    def __len__(self):
        return self.current_storage_len

    def add(self, data):
        if self._next_idx >= self.current_storage_len:
            self.current_storage_len += 1
        self.actions[self._next_idx] = data[0]
        self.rewards[self._next_idx] = data[1]
        self.states[self._next_idx] = data[2]
        self.dones[self._next_idx] = data[3]
        self.ishumans[self._next_idx] = data[4]
        if(self.IS_HUMAN):
            self.r1_rewards[self._next_idx] = data[5]
            self.step_in_ep[self._next_idx] = data[6]
            if(self.USE_DFS):
                self.r2_rewards[self._next_idx] = data[7]
        if(not self.IS_HUMAN):
            self.life_loss[self._next_idx] = data[5]

        #self._storage[self._next_idx] = data
        idx = self._next_idx
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        self._next_idx = (self._next_idx + 1) % self._maxsize



    def _encode_sample(self, idxes):
        b_memory = np.empty((idxes.__len__(), 10), dtype=object)
        # experience = []
        batch_idx = 0
        for i in idxes:
            data = self.get_data(i)
            b_memory[batch_idx] = data
            batch_idx += 1
        return b_memory

    def get_data(self, idx):
        if (self.IS_HUMAN):
            if (self.SKIP_R1 == 1):
                self.get_human_data_noskip(idx)
            if (self.USE_DFS):
                return self.get_human_data_dfs(idx)
            else:
                return self.get_human_data(idx)
        else:
            return self.get_actor_data(idx)

    def get_actor_data(self, idx):
        action = self.actions[idx % self._maxsize]
        reward = self.rewards[idx % self._maxsize]
        prestates = self.get_actor_state(idx - 1)
        poststates = self.get_actor_state(idx)

        done = self.dones[idx % self._maxsize] or self.life_loss[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]

        n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_data(idx)
        return [prestates, action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done,
                actual_n]

    def get_actor_state(self, idx):
        return np.asarray([self.states[(idx - 3) % self._maxsize][0],
                           self.states[(idx - 2) % self._maxsize][0],
                           self.states[(idx - 1) % self._maxsize][0],
                           self.states[idx % self._maxsize][0]])

    def get_nstep_data(self, idx):
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        for i in range(self.n_step):
            if (not (self.dones[(idx + i) % self._maxsize] or self.life_loss[(idx + i) % self._maxsize])):
                n_step_reward += self.rewards[(idx + i) % self._maxsize] * (Config.GAMMA ** i)
            else:
                n_step_done = True
                n_step_reward -= self.rewards[(idx + (i - 1)) % self._maxsize] * (Config.GAMMA ** (i - 1))
                actual_n = i - 1
                break
        if (n_step_done == False):
            if (not (self.dones[(idx + self.n_step) % self._maxsize] or self.life_loss[
                    (idx + self.n_step) % self._maxsize])):
                actual_n = self.n_step
            else:
                n_step_done = True
                n_step_reward -= self.rewards[(idx + (self.n_step - 1)) % self._maxsize] * (
                Config.GAMMA ** (self.n_step - 1))
                actual_n = self.n_step - 1
        n_step_state = self.get_actor_state(idx + actual_n)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_human_data_noskip(self, idx):
        action = self.actions[idx % self._maxsize]
        reward = self.rewards[idx % self._maxsize]
        prestates = self.get_actor_state(idx - 1)
        poststates = self.get_actor_state(idx)

        done = self.dones[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]

        n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_human_data_noskip(idx)
        return [prestates, action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done,
                actual_n]

    def get_human_data(self, idx):
        action = self.actions[idx % self._maxsize]
        reward = self.r1_rewards[(idx) % self._maxsize]
        prestates = self.get_human_states((idx - self.SKIP_R1))
        poststates = self.get_human_states(idx)

        done = self.dones[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]

        n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_human_data(idx)

        return [prestates, action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done,
                actual_n]

    def get_nstep_human_data(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        for i in range(self.n_step):
            current_ep_count = self.step_in_ep[(idx + (i * self.SKIP_R1)) % self._maxsize]
            if (ep_count <= current_ep_count):
                n_step_reward += self.r1_rewards[(idx + i * self.SKIP_R1) % self._maxsize] * (Config.GAMMA ** i)
            else:
                n_step_done = True
                n_step_reward -= self.r1_rewards[(idx + (i - 1) * self.SKIP_R1) % self._maxsize] * (
                Config.GAMMA ** (i - 1))
                actual_n = i - 1
                break
        if (n_step_done == False):
            current_ep_count = self.step_in_ep[(idx + ((self.n_step) * self.SKIP_R1)) % self._maxsize]
            if (ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                n_step_reward -= self.r1_rewards[(idx + (self.n_step - 1) * self.SKIP_R1) % self._maxsize] * (
                Config.GAMMA ** (self.n_step - 1))
                actual_n = self.n_step - 1
        n_step_state = self.get_human_states((idx + actual_n * self.SKIP_R1) % self._maxsize)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_nstep_human_data_noskip(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False

        actual_n = 0
        for i in range(self.n_step):
            current_ep_count = self.step_in_ep[(idx + (i * self.SKIP_R1)) % self._maxsize]
            if (ep_count <= current_ep_count):
                reward = self.rewards[(idx + (i * self.SKIP_R1)) % self._maxsize] * (Config.GAMMA ** i)
                n_step_reward += reward
            else:
                n_step_done = True
                reward = self.rewards[(idx + ((i - 1) * self.SKIP_R1)) % self._maxsize] * (Config.GAMMA ** (i - 1))
                n_step_reward -= reward
                actual_n = i - 1
                break
        if (n_step_done == False):
            current_ep_count = self.step_in_ep[(idx + ((self.n_step) * self.SKIP_R1)) % self._maxsize]
            if (ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                reward = self.rewards[(idx + ((self.n_step - 1) * self.SKIP_R1)) % self._maxsize] * (
                Config.GAMMA ** (self.n_step - 1))
                n_step_reward -= reward
                actual_n = self.n_step - 1
        n_step_state = self.get_actor_state((idx + actual_n * self.SKIP_R1) % self._maxsize)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_human_data_dfs(self, idx):
        action = self.actions[idx % self._maxsize]

        if (self.actions[(idx - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
            prestates = self.get_human_states_dfs(idx - self.SKIP_R2)
            reward = self.r2_rewards[(idx) % self._maxsize]
        else:
            prestates = self.get_human_states_dfs((idx - self.SKIP_R1))
            reward = self.r1_rewards[(idx) % self._maxsize]
        poststates = self.get_human_states(idx)

        done = self.dones[idx % self._maxsize]
        ishuman = self.ishumans[idx % self._maxsize]
        if (self.DYNAMIC_N_Q_STEP):
            n_step_reward, n_step_done, actual_n, n_step_state = self.get_dynamic_nstep_human_data_dfs(idx)
        else:
            n_step_reward, n_step_done, actual_n, n_step_state = self.get_nstep_human_data_dfs(idx)

        return [prestates, action, reward, poststates, done, ishuman, n_step_reward, n_step_state, n_step_done,
                actual_n]

    def get_human_states_dfs(self, idx):
        states = np.zeros((Config.HISTORY_LENGTH, 84, 84), dtype=np.uint8)
        index = idx
        state = self.states[(index) % self._maxsize][0]
        states[Config.HISTORY_LENGTH - 1] = state
        for i in range(Config.HISTORY_LENGTH - 1):
            if (self.actions[(index - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
                state = self.states[(index - self.SKIP_R2) % self._maxsize][0]
                states[Config.HISTORY_LENGTH - 2 - i] = state
                index = index - self.SKIP_R2
            else:
                state = self.states[(index - self.SKIP_R1) % self._maxsize][0]
                states[Config.HISTORY_LENGTH - 2 - i] = state
                index = index - self.SKIP_R1
        return states

    def get_dynamic_nstep_human_data_dfs(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        indexes = []
        sum_dynamic_n = 0
        if (self.actions[(idx - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
            index = idx - self.SKIP_R2
        else:
            index = idx - self.SKIP_R1
        indexes.append(index)
        index = idx
        indexes.append(index)
        action = self.actions[index % self._maxsize]

        for i in range(self.n_step):

            current_ep_count = self.step_in_ep[(indexes[i + 1]) % self._maxsize]
            if (ep_count <= current_ep_count):
                if (abs(indexes[i + 1] - indexes[i]) == self.SKIP_R2):
                    n_step_reward += self.r2_rewards[(indexes[i + 1]) % self._maxsize] * (
                    Config.GAMMA ** (sum_dynamic_n))
                else:
                    n_step_reward += self.r1_rewards[(indexes[i + 1]) % self._maxsize] * (Config.GAMMA ** sum_dynamic_n)
            else:
                n_step_done = True
                if (abs(indexes[i - 1] - indexes[i]) == self.SKIP_R2):
                    sum_dynamic_n = sum_dynamic_n - self.DYNAMIC_N
                    n_step_reward -= self.r2_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n))
                else:
                    sum_dynamic_n = sum_dynamic_n - 1
                    n_step_reward -= self.r1_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (sum_dynamic_n))
                actual_n = i - 1
                break
            if (action >= self.ACTION_NUM):
                index = index + self.SKIP_R2
                indexes.append(index)
                sum_dynamic_n = sum_dynamic_n + self.DYNAMIC_N

            else:
                index = index + self.SKIP_R1
                indexes.append(index)
                sum_dynamic_n = sum_dynamic_n + 1
            action = self.actions[index % self._maxsize]

        if (n_step_done == False):
            current_ep_count = self.step_in_ep[indexes[self.n_step + 1] % self._maxsize]

            if (ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                if (abs(indexes[self.n_step - 1] - indexes[self.n_step]) == self.SKIP_R2):
                    sum_dynamic_n -= self.DYNAMIC_N
                    n_step_reward -= self.r2_rewards[indexes[self.n_step] % self._maxsize] * (
                    Config.GAMMA ** (sum_dynamic_n))
                else:
                    sum_dynamic_n -= 1
                    n_step_reward -= self.r1_rewards[indexes[self.n_step] % self._maxsize] * (
                    Config.GAMMA ** (sum_dynamic_n))
                actual_n = self.n_step - 1
        n_step_state = self.get_human_states_dfs(indexes[actual_n + 1] % self._maxsize)
        return n_step_reward, n_step_done, sum_dynamic_n, n_step_state

    def get_nstep_human_data_dfs(self, idx):
        ep_count = self.step_in_ep[idx % self._maxsize]
        n_step_reward = 0
        n_step_done = False
        actual_n = 0
        indexes = []

        if (self.actions[(idx - self.SKIP_R2) % self._maxsize] >= self.ACTION_NUM):
            index = idx - self.SKIP_R2
        else:
            index = idx - self.SKIP_R1
        indexes.append(index)
        index = idx
        indexes.append(index)
        action = self.actions[index % self._maxsize]

        for i in range(self.n_step):

            current_ep_count = self.step_in_ep[(indexes[i + 1]) % self._maxsize]
            if (ep_count <= current_ep_count):
                if (abs(indexes[i + 1] - indexes[i]) == self.SKIP_R2):
                    n_step_reward += self.r2_rewards[(indexes[i + 1]) % self._maxsize] * (Config.GAMMA ** (i))
                else:
                    n_step_reward += self.r1_rewards[(indexes[i + 1]) % self._maxsize] * (Config.GAMMA ** i)
            else:
                n_step_done = True
                if (abs(indexes[i - 1] - indexes[i]) == self.SKIP_R2):
                    n_step_reward -= self.r2_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (i - 1))
                else:
                    n_step_reward -= self.r1_rewards[(indexes[i]) % self._maxsize] * (Config.GAMMA ** (i - 1))
                actual_n = i - 1
                break
            if (action >= self.ACTION_NUM):
                index = index + self.SKIP_R2
                indexes.append(index)

            else:
                index = index + self.SKIP_R1
                indexes.append(index)
            action = self.actions[index % self._maxsize]

        if (n_step_done == False):
            current_ep_count = self.step_in_ep[indexes[self.n_step + 1] % self._maxsize]

            if (ep_count < current_ep_count):
                actual_n = self.n_step
            else:
                n_step_done = True
                if (abs(indexes[self.n_step - 1] - indexes[self.n_step]) == self.SKIP_R2):
                    n_step_reward -= self.r2_rewards[indexes[self.n_step] % self._maxsize] * (
                    Config.GAMMA ** (self.n_step - 1))
                else:
                    n_step_reward -= self.r1_rewards[indexes[self.n_step] % self._maxsize] * (
                    Config.GAMMA ** (self.n_step - 1))
                actual_n = self.n_step - 1
        n_step_state = self.get_human_states_dfs(indexes[actual_n + 1] % self._maxsize)
        return n_step_reward, n_step_done, actual_n, n_step_state

    def get_human_states(self, idx):
        return np.asarray([self.states[(idx - self.SKIP_R1 * 3) % self._maxsize][0],
                           self.states[(idx - self.SKIP_R1 * 2) % self._maxsize][0],
                           self.states[(idx - self.SKIP_R1 * 1) % self._maxsize][0],
                           self.states[idx % self._maxsize][0]])

    def _sample_proportional(self, batch_size):
        if(self._alpha == 0):
            res = [random.randint(0, self.current_storage_len - 1) for _ in range(batch_size)]
        else :
            res = []
            p_total = self._it_sum.sum(0, self.current_storage_len - 1)
            every_range_len = p_total / batch_size
            for i in range(batch_size):
                # TODO(szymon): should we ensure no repeats?
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
                res.append(idx)
        return res
    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        if(self._alpha > 0):
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * self.current_storage_len) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * self.current_storage_len) ** (-beta)
                weights.append([weight / max_weight])
            weights = np.array(weights)
        else:
            for idx in idxes:
                weights.append([1.0])
            weights = np.array(weights)

        encoded_sample = self._encode_sample(idxes)

        return idxes, encoded_sample, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.current_storage_len
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def getStorageLength(self):
        return self.current_storage_len