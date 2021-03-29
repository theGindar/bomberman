import os
import pickle
import random

import numpy as np
import torch
import math
from .shared import device, EPS_END, EPS_START, EPS_DECAY
from .utils import state_to_features
from .train import policy_net
import torch.nn.functional as F
import agent_code.my_agent_ADRQN.global_model_variables
import time


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
    self.steps_count = 0
    if self.train:
        print('train is true')
        #setup_training(self)

    #setup_training(self)
    #self.policy_net = policy_net

def act(self, game_state: dict) -> str:
    #start = time.time()
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state['step'] == 1:
        agent_code.my_agent_ADRQN.global_model_variables.last_action = 4

    features = state_to_features(game_state)
    #print(f'steps_done in act: {self.steps_done}')
    
    sample = random.random()
    random_threshold = 0.05
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.steps_count / EPS_DECAY)
    self.steps_count += 1
    q_values, agent_code.my_agent_ADRQN.global_model_variables.hidden_s = \
        policy_net.act(torch.unsqueeze(features.to(device), 2),
                       F.one_hot(torch.tensor(agent_code.my_agent_ADRQN.global_model_variables.last_action), 6).view(1,1,-1).float().to(device),
                       hidden=agent_code.my_agent_ADRQN.global_model_variables.hidden_s)

    #for i in range(6):
    action = torch.argmax(q_values).item()
        #break
        #if check_validity(game_state, action):
        #    break
        #else:
        #    q_values[0][0][torch.argmax(q_values)] = 0.0

    #print(f'Threshold: {eps_threshold}')
    if (sample < random_threshold) and check_validity(game_state, torch.tensor([[5]])):
        #action = torch.tensor([[np.random.choice(ACTION_NUMBERS, p=[.175, .175, .175, .175, .1, .2])]], device=device, dtype=torch.long)
        action = torch.tensor([[5]])
        #action = torch.tensor([[np.random.choice(ACTION_NUMBERS, p=[.2, .2, .2, .2, .1, .1])]], device=device, dtype=torch.long)

    agent_code.my_agent_ADRQN.global_model_variables.last_action = action
        # action = policy_net(features.to(device)).max(1)[1].view(1, 1)

    #print(f'steps done: {self.steps_done}')

    
    #print(f'action chosen: {action}')
    # todo Exploration vs exploitation
    #random_prob = .1
    #if self.train and random.random() < random_prob:
    #    self.logger.debug("Choosing action purely at random.")
    #    # 80%: walk in any direction. 10% wait. 10% bomb.
    #    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    #print(f'execution time: {end-start}')
    #self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)
    if game_state['step'] % 50 == 0:
        # todo problem: wenn weniger als 400 steps wird es nicht zurückgesetzt...
        self.steps_count = 0
        #agent_code.my_agent_ADRQN.global_model_variables.last_action = 4
    #end = time.time()
    #print(end-start)
    return ACTIONS[action]


def check_validity(game_state, action):
    valid = True
    can_place_bomb = game_state['self'][2]
    agent_position = game_state['self'][3]

    if action == 0:
        # up
        if game_state['field'][agent_position[0], agent_position[1]-1] != 0:
            valid = False
    elif action == 1:
        # right
        if game_state['field'][agent_position[0]+1, agent_position[1]] != 0:
            valid = False
    elif action == 2:
        # down
        if game_state['field'][agent_position[0], agent_position[1]+1] != 0:
            valid = False
    elif action == 3:
        # left
        if game_state['field'][agent_position[0]-1, agent_position[1]] != 0:
            valid = False
    elif action == 5:
        # bomb
        if not can_place_bomb:
            valid = False
        if agent_position == (1,1) or agent_position == (1,15) or agent_position == (15,15) or agent_position == (15,1):
            valid = False
    return valid