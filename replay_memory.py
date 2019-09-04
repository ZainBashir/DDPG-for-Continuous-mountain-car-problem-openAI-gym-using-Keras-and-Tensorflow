import random
from collections import deque
import numpy as np

#Class that implements memory structure which stores experiences. Also implements methods to store, rerieve
#and count experiences.
class ReplayMemory:
    # initialize the buffer as a python deque structure of the required size
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    #Push the experience tuple to the memory in a FIFO fashion
    def append(self,state, action, reward, next_state, next_action, done):
        self.memory.append([state,action,reward,next_state, next_action, done])

    #Function that samples the memory randomly to retrieve a batch of experiences. The number of samples
    #retrieved is defined by the batch_size
    def sample(self,batch_size):
        #Sample randomly
        batch = random.sample(self.memory,batch_size)

        #Lists to store each component of the experience separately
        current_states = []
        rewards = []
        actions = []
        next_states = []
        next_actions = []
        dones = []

        #loop through each experience sample in the batch and unpack components of the sample
        for sample in batch:
            state, action, reward, next_state, next_action, done = sample
            current_states.append(state)
            rewards.append(reward)
            actions.append(action)
            next_states.append(next_state)
            next_actions.append(next_action)
            dones.append(done)
        #Convert from list to array to perform later functions
        current_states = np.array(current_states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        next_actions = np.array(next_actions)

        #Reshape
        rewards = np.array(rewards).reshape((batch_size,1))
        dones = np.array(dones).reshape((batch_size,1))

        return [current_states,actions, rewards, next_states, next_actions, dones]

    #Return the number of samples stored in the memory at any given time
    def count(self):
        return len(self.memory)