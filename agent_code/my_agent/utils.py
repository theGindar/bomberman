import torch
import pickle
import numpy as np
from datetime import datetime

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
        print('GAMESTATE IS NONE')
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

    return current_state

def save_rewards_to_file(rewards_list):
    current_datetime = datetime.now()
    with open(f'./rewards/rewards_{str(current_datetime)}.pkl', 'wb') as f:
       pickle.dump(rewards_list, f)

