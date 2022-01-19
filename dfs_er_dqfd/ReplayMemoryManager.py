from Config import Config
import numpy as np
class ReplayMemoryManager(object):
    def __init__(self, humanMemory, actorMemory, DECAY_LENGTH=1000000, LINAER_DECAY=True, LOGARITHM_FAST_DECAY=False, LOGARITHM_DECAY=False, CONSTANT=False, ONLY_ACTOR=False, SINGLE_REPLAY_DECAY=False, PRIORITY_REPLAY=True, USE_SINGLE_REPLAY=False):

        self.BATCH_SIZE = Config.BATCH_SIZE
        self.DECAY_LENGTH = DECAY_LENGTH
        self.FINAL_HUMAN_RATIO = Config.FINAL_HUMAN_RATIO
        self.LINEAR_DECAY = LINAER_DECAY
        self.LOGARITHM_FAST_DECAY = LOGARITHM_FAST_DECAY
        self.LOGARITHM_DECAY = LOGARITHM_DECAY
        self.CONSTANT = CONSTANT
        self.ONLY_ACTOR = ONLY_ACTOR
        self.SINGLE_REPLAY_DEACY = SINGLE_REPLAY_DECAY
        self.PRIORITY_REPLAY = PRIORITY_REPLAY

        self.USE_SINGLE_REPLAY = USE_SINGLE_REPLAY

        self.HUMAN_EPSILON = Config.HUMAN_EPSILON
        self.ACTOR_EPSILON = Config.ACTOR_EPSILON
        self.config = Config
        self.actorMemory = actorMemory
        self.humanMemory = humanMemory
        self.BETA = Config.BETA
        self.beta_per_step = 1/(100000000)
    def add(self, experience):
        is_demo = experience[4]
        if(is_demo == 1):
            self.humanMemory.add(experience)
        elif(is_demo == 0):
            self.actorMemory.add(experience)

    def merge_data_by_priority(self, actor_batch):
        self.BETA += self.beta_per_step
        if(actor_batch == 0):
            return self.humanMemory.sample(self.BATCH_SIZE, self.BETA)
        elif(actor_batch == self.BATCH_SIZE):
            return self.actorMemory.sample(self.BATCH_SIZE, self.BETA)
        else:
            idxes_human, datas_human, weights_human = self.humanMemory.sample(self.BATCH_SIZE - actor_batch, self.BETA)
            idxes_actor, datas_actor, weights_actor = self.actorMemory.sample(actor_batch, self.BETA)

            idxes = np.concatenate((idxes_human, idxes_actor), axis=0)
            datas = np.concatenate((datas_human, datas_actor), axis=0)
            weights = np.concatenate((weights_human, weights_actor), axis=0)
            return idxes, datas, weights

    def merge_data(self, actor_batch):
        if(actor_batch == 0):
            return self.humanMemory.sample(self.BATCH_SIZE)
        elif(actor_batch == self.BATCH_SIZE):
            return self.actorMemory.sample(self.BATCH_SIZE)
        else:
            datas_human = self.humanMemory.sample(self.BATCH_SIZE - actor_batch)
            datas_actor = self.actorMemory.sample(actor_batch)

            #print(self.actorMemory.getStorageLength())
            datas = np.concatenate((datas_human, datas_actor), axis=0)
            #print(datas_human.__len__())
            return datas

    def get_merged_data(self, actor_batch):
        if (self.PRIORITY_REPLAY == True):
            return self.merge_data_by_priority(actor_batch)
        else:
            return self.merge_data(actor_batch)

    def get_actor_batch_size_linear(self, trainstep, pretrainstep):
        rlstep = trainstep - pretrainstep
        actor_batch = int((self.BATCH_SIZE / self.DECAY_LENGTH) * rlstep)
        if(actor_batch > self.BATCH_SIZE *(1- self.FINAL_HUMAN_RATIO)):
            actor_batch =  int(self.BATCH_SIZE *(1- self.FINAL_HUMAN_RATIO))
        return actor_batch

    def get_actor_batch_size_logarithm_slow(self, trainstep, pretrainstep):
        rlstep = trainstep - pretrainstep
        decayed_batch = self.BATCH_SIZE * 0.96 ** (rlstep / 10000)
        actor_batch = int(self.BATCH_SIZE - decayed_batch)
        if(actor_batch > self.BATCH_SIZE *(1- self.FINAL_HUMAN_RATIO)):
            actor_batch =  int(self.BATCH_SIZE *(1- self.FINAL_HUMAN_RATIO))
        return actor_batch

    def get_actor_batch_size_logarithm_fast(self, trainstep, pretrainstep):
        rlstep = trainstep - pretrainstep
        decayed_batch = self.BATCH_SIZE * 0.96 ** (rlstep / 5000)
        actor_batch = int(self.BATCH_SIZE - decayed_batch)
        if(actor_batch > self.BATCH_SIZE *(1- self.FINAL_HUMAN_RATIO)):
            actor_batch =  int(self.BATCH_SIZE *(1- self.FINAL_HUMAN_RATIO))
        return actor_batch

    def get_actor_batch_size_single_replay(self, trainstep, pretrainstep):
        rlstep = trainstep - pretrainstep
        j=0
        if(rlstep < 50000):
            j = rlstep
        else:
            j = 50000
        actor_batch = int(32 * (j / (12500 + j)))
        return actor_batch

    def get_actor_batch_size_constant(self, trainstep, pretrainstep):
        actor_batch = int(self.BATCH_SIZE * 0.75)
        return actor_batch

    def get_actor_batch_size_onlyactor(self, trainstep, pretrainstep):
        actor_batch = int(self.BATCH_SIZE)
        return actor_batch

    def sample(self, trainstep, pretrainstep):
        #print(trainstep)
        if(self.USE_SINGLE_REPLAY):
            return self.get_merged_data(0)
        else:
            if(trainstep <= pretrainstep):
                return self.get_merged_data(0)
            else:
                if(self.LINEAR_DECAY == True):
                    actor_batch = self.get_actor_batch_size_linear(trainstep,  pretrainstep)
                    return self.get_merged_data(actor_batch)
                elif(self.LOGARITHM_DECAY == True):
                    actor_batch = self.get_actor_batch_size_logarithm_slow(trainstep,  pretrainstep)
                    return self.get_merged_data(actor_batch)
                elif(self.LOGARITHM_FAST_DECAY== True):
                    actor_batch = self.get_actor_batch_size_logarithm_fast(trainstep,  pretrainstep)
                    return self.get_merged_data(actor_batch)
                elif(self.CONSTANT == True):
                    actor_batch = self.get_actor_batch_size_constant(trainstep,  pretrainstep)
                    return self.get_merged_data(actor_batch)
                elif(self.ONLY_ACTOR == True):
                    actor_batch = self.get_actor_batch_size_onlyactor(trainstep,  pretrainstep)
                    return self.get_merged_data(actor_batch)
                elif(self.SINGLE_REPLAY_DEACY == True):
                    actor_batch = self.get_actor_batch_size_single_replay(trainstep, pretrainstep)
                    return self.get_merged_data(actor_batch)

    def update_priorities(self, tree_idxes, abs_errors, demo_data):
        if(self.PRIORITY_REPLAY == True):
            human_idxes, human_priority = [], []
            actor_idxes, actor_priority  = [], []
            human_idxes = np.extract(demo_data, tree_idxes)
            human_priority = np.extract(demo_data, abs_errors)
            actor_idxes = np.extract(1-demo_data, tree_idxes)
            actor_priority = np.extract(1-demo_data, abs_errors)
            actor_priority += self.ACTOR_EPSILON
            human_priority += self.HUMAN_EPSILON

            #actor_priority = np.minimum(actor_priority, 1.)
            #human_priority = np.minimum(human_priority, 1.)

            if(human_idxes.__len__() != 0):
                self.humanMemory.update_priorities(human_idxes, human_priority)
            if (actor_idxes.__len__() != 0):
                self.actorMemory.update_priorities(actor_idxes, actor_priority)

    def isHumanFull(self):
        if(self.USE_SINGLE_REPLAY):
            if(self.humanMemory.getStorageLength() >= self.humanMemory.permanent_size):
                return True
            else:
                return False
        if(self.humanMemory.getStorageLength() >= self.humanMemory._maxsize):
            return True
        elif(self.humanMemory.getStorageLength() < self.humanMemory._maxsize):
            return False

    def isActorFull(self):
        if(self.actorMemory.getStorageLength() >= self.actorMemory._maxsize):
            return True
        elif(self.actorMemory.getStorageLength() < self.actorMemory._maxsize):
            return False

    def getActorMemoryLength(self):
        return self.actorMemory.getStorageLength()

    def getHumanMemoryLength(self):
        return self.humanMemory.getStorageLength()