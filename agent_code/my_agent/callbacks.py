import os
import pickle
import random

import numpy as np
import torch
import math
from .shared import device, EPS_END, EPS_START, EPS_DECAY
from .utils import state_to_features
from .train import policy_net


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_NUMBERS = [0,1,2,3,4,5]

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.steps_done = 0
    if self.train:
        print('train is true')
        #setup_training(self)

    #setup_training(self)
    #self.policy_net = policy_net

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.steps_done / EPS_DECAY)
    self.steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(features.to(device)).max(1)[1].view(1, 1)
            
            # print(f'action chosen: {action}')

            
            
    else:
        # action = torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)
        #action = policy_net(features).max(1)[1].view(1, 1)
        #action = torch.tensor([[np.random.choice(ACTION_NUMBERS, p=[.225, .225, .225, .225, .1, .0])]], device=device, dtype=torch.long)
        
        #action = torch.tensor([[np.random.choice(ACTION_NUMBERS, p=[.2, .2, .2, .2, .1, .1])]], device=device, dtype=torch.long)
        action = torch.tensor([[np.random.choice(ACTION_NUMBERS, p=[.2, .2, .2, .2, .15, .05])]], device=device, dtype=torch.long)

        #action = torch.tensor([[np.random.choice(ACTION_NUMBERS, p=[0, 0, 0, 0, 0, 1])]], device=device, dtype=torch.long)

        #action = policy_net(features.to(device)).max(1)[1].view(1, 1)

    #print(f'steps done: {self.steps_done}')
    
    
    
    #print(f'action chosen: {action.item()}')
    # todo Exploration vs exploitation
    #random_prob = .1
    #if self.train and random.random() < random_prob:
    #    self.logger.debug("Choosing action purely at random.")
    #    # 80%: walk in any direction. 10% wait. 10% bomb.
    #    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    #self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    if self.steps_done == 400:
        self.steps_done = 0
    return ACTIONS[action.item()]




