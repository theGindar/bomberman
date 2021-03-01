import os
import pickle
import random

import numpy as np
import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    vector_shape = state_to_features(game_state)

    self.logger.debug(f'shape of current_state: {vector_shape}')
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # convert game state to input tensor for the model
    current_state = torch.zeros((1, 6, 17, 17))
    current_state[0, 0] = torch.from_numpy(game_state['field'])

    # set agents that can place a bomb to 1, otherwise to -1
    if game_state['self'][2] == True: 
        current_state[0, 1, game_state['self'][3][0], game_state['self'][3][1]] = 1
    else:
        current_state[0, 1, game_state['self'][3][0], game_state['self'][3][1]] = -1

    for other in game_state['others']:
        if other[2] == True:
            current_state[0, 2, other[3][0], other[3][1]] = 1
        else:
            current_state[0, 2, other[3][0], other[3][1]] = -1

    for bomb in game_state['bombs']:
        current_state[0, 3, bomb[0][0], bomb[0][1]] = bomb[1]

    current_state[0, 4] = torch.from_numpy(game_state['explosion_map'])

    for coin in game_state['coins']:
        current_state[0, 5, coin[0], coin[1]] = 1
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    return current_state.reshape(-1).size()
