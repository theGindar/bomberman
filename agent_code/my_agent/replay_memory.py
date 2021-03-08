import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_connected_timeseries(self, batch_size):
        """Returns a subsequence of size batch_size from the replay memory."""
        max_idx = 0
        if batch_size <= len(self.memory):
            max_idx = len(self.memory) - batch_size
        idx = random.randint(0, max_idx)
        return self.memory[max_idx:(max_idx + batch_size)]

    def __len__(self):
        return len(self.memory)