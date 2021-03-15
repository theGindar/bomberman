import random
import numpy as np
import  torch
from .shared import device
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, aoarod):
      if self.counter < self.max_storage-1:
          self.counter +=1
          if self.filled < self.max_storage-1:
            self.filled += 1
      else:
          self.counter = 0
      self.storage[self.counter] = aoarod

    def sample(self, batch_size):
        #Returns tensors of sizes (batch_size, seq_len, *) where * depends on action/observation/return/done
        seq_len = self.sample_length
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            if self.filled - seq_len < 0 :
                raise Exception("Reduce seq_len or increase exploration at start.")
            start_idx = np.random.randint(self.filled-seq_len)
            last_act, last_obs, act, rew, obs, done = zip(*self.storage[start_idx:start_idx+seq_len])
            last_actions.append(list(last_act))
            last_observations.append(torch.cat(last_obs))
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append(torch.cat(list(obs)))
            dones.append(list(done))
        return torch.tensor(last_actions).to(device), \
               torch.stack(last_observations).permute(0, 2, 1, 3, 4), \
               torch.tensor(actions).to(device), \
               torch.tensor(rewards).float().to(device), \
               torch.stack(observations).permute(0, 2, 1, 3, 4), \
               torch.tensor(dones).to(device)

    def __len__(self):
        if self.filled < 0:
            return 0
        else:
            return self.filled