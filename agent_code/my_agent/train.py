import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import numpy as np

import events as e
import copy
from .model import Model
from .utils import state_to_features, save_rewards_to_file, save_loss_to_file
from .replay_memory import ReplayMemory
import torch
import torch.optim as optim
import torch.nn.functional as F
from .shared import device, EPS_START, EPS_END, EPS_DECAY


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = { 'UP': 0, 
            'RIGHT': 1, 
            'DOWN': 2, 
            'LEFT': 3, 
            'WAIT': 4,
            'BOMB': 5}

# Hyper parameters -- DO modify
#TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...!!!!!!!!!!!!!!!!!!!!!!!

BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10
NUM_EPISODES = 1000
LEARNING_RATE = 0.0001

target_net = Model().to(device)
policy_net = Model().to(device)

if len(os.listdir("./agent_code/my_agent/saved_models")) != 0:
    print('loading existing model...')
    policy_net.load_state_dict(torch.load("./agent_code/my_agent/saved_models/krasses_model.pt", map_location=torch.device('cpu')))
    #policy_net.load_state_dict(torch.load("./agent_code/my_agent/saved_models/krasses_model.pt"))
    policy_net.eval()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    print('setup training called')
    self.steps_done = 0
    self.current_episode_num = 1
    self.total_reward = 0

    #self.optimizer = optim.RMSprop(policy_net.parameters())
    self.memory = ReplayMemory(300000)
    self.total_reward_history = []
    #self.loss_history = []
    self.positions = []
    self.n_destroyed_crates = 0
    self.is_in_bomb_range = False



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    min_coin_distance = calc_coin_distance(new_game_state)
    # if calc_position_change(self, new_game_state):
    #    events.append('REPEATS_STEPS')
    if calc_is_new_position(self, new_game_state):
        events.append('NEW_POSITION_EXPLORED')

    # set reward if bomb is dropped near crates
    if 'BOMB_DROPPED' in events:
        self.n_destroyed_crates = crates_destroyed(self, new_game_state)

    # set variable is_in_bomb_range
    if in_bomb_range(self, new_game_state):
        events.append('IN_BOMB_RANGE')

    reward = reward_from_events(self, events, min_coin_distance, new_game_state)

    
    self.total_reward += reward
    reward = torch.tensor([reward], device=device)
    if  state_to_features(old_game_state) != None:
        self.memory.push(state_to_features(old_game_state).to(device), self_action, state_to_features(new_game_state).to(device), reward)

    self.positions.append(new_game_state['self'][3])

    self.old_game_state = old_game_state
    optimize_model(self)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #reward = reward_from_events(self, events)
    #self.total_reward += reward
    #self.total_reward_tensor = torch.tensor([self.total_reward], device=device)

    #self.memory.push(state_to_features(self.old_game_state), last_action, state_to_features(last_game_state), self.total_reward_tensor)

    #optimize_model(self)


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    if self.current_episode_num % TARGET_UPDATE == 0:
        print('update target_net')
        target_net.load_state_dict(policy_net.state_dict())

    print(f'finished episode {self.current_episode_num}')
    self.current_episode_num += 1
    
    self.total_reward_history.append(self.total_reward)
    if self.current_episode_num == NUM_EPISODES:
        torch.save(target_net.state_dict(), "./saved_models/krasses_model.pt")
        save_rewards_to_file(self.total_reward_history)
        #save_loss_to_file(self.loss_history)
    self.total_reward = 0

    #print(f'Positions: {len(self.positions)}')
    self.positions = []


def reward_from_events(self, events: List[str], distance_coin, game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.

    events: list of events
    distance_coin: distance of agent to the nearest coin
    """
    game_rewards = {
        e.COIN_COLLECTED: 700,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -35,
        e.MOVED_DOWN: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.WAITED: 0,
        e.BOMB_DROPPED: 0,
        e.IN_BOMB_RANGE: -10,
        e.KILLED_SELF: -1000
    }
    for i in events:
        if i == e.REPEATS_STEPS:
            #("step repeated!")
            pass
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    if len(game_state['coins']) != 0:
        reward_sum += int(5 - distance_coin*5)

    #set reward for destroyed crates
    reward_sum += self.n_destroyed_crates*200
    self.n_destroyed_coins = 0
    #print(f'Number of destroyed crates: {self.n_destroyed_crates}')
    print(f'Reward: {reward_sum}')

    """
    # normalize the calculated reward
    max_reward = 0
    min_reward = 0
    for key, value in game_rewards.items():
        if value > 0:
            max_reward += value
        else:
            min_reward -= value

    reward_sum = (reward_sum + min_reward) / (max_reward+min_reward+100+3600)
    #print(f'Reward {reward_sum}')
    """
    return reward_sum

def calc_coin_distance(game_state: dict):
    # maximum distance an agent could have to a coin
    max_distance = 21.5
    # minimum distance an agent could (theoretically) have to a coin
    min_distance = 0

    """calculate distance of agent to next coin (a value between 0 and 1)"""
    min_coin_distance = float(10000)
    agent_position = np.asarray(game_state['self'][3])
    for coin in game_state['coins']:
        dist = np.linalg.norm(agent_position-np.asarray(coin))
        if min_coin_distance > dist:
            min_coin_distance = dist

    # normalize to a value between 0 and 1
    return (min_coin_distance - min_distance) / (max_distance - min_distance)

def calc_position_change(self, game_state: dict):
    """
    current_position = position for respective step
    positions = List with all positions for each step (x,y tuple)

    Compare if one of the last three positions is the same as the current position (the agent repeats its steps).
    If so: Set true and therefore give a negative reward.
    """
    current_position = game_state['self'][3]
    #print(f'Current Position: {current_position}')
    while len(self.positions) > 3:
        self.positions.pop(0)

        if current_position in self.positions:
            return True
        else:
            return False

def calc_is_new_position(self, game_state: dict):
    """
    return True, if the current position of the agent was not explored before
    """
    current_position = game_state['self'][3]
    if current_position in self.positions:
        return False
    else:
        return True

def crates_destroyed(self, game_state:dict):
    """

    returns the number of crates, which will be destroyed by a bomb

    iterate through every line in which the bomb will be destroyed.
    """

    bomb_position_x = game_state['self'][3][0]
    bomb_position_y = game_state['self'][3][1]
    n_crates = 0

    for i in range(3):
        if bomb_position_x-i-1 >= 0:
            if game_state['field'][bomb_position_x-i-1][bomb_position_y] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x-i-1][bomb_position_y] == -1:
                break

    for i in range(3):
        if bomb_position_x+i+1 <= 16:
            if game_state['field'][bomb_position_x+i+1][bomb_position_y] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x+i+1][bomb_position_y] == -1:
                break

    for i in range(3):
        if bomb_position_y-i-1 >= 0:
            if game_state['field'][bomb_position_x][bomb_position_y-i-1] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x][bomb_position_y-i-1] == -1:
                break

    for i in range(3):
        if bomb_position_y+i+1 <= 16:
            if game_state['field'][bomb_position_x][bomb_position_y+i+1] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x][bomb_position_y+i+1] == -1:
                break

    return n_crates

def in_bomb_range(self, game_state: dict):
    """

    returns true, if the agent is in the bomb explosion radius

    """
    is_in_bomb_range = False
    agent_position = game_state['self'][3]
    agent_position = list(agent_position)

    for bomb in game_state['bombs']:
        if agent_position == list(bomb[0]):
            is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[0]-i-1 >= 0:
                agent_search[0] = agent_position[0]-i-1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[0]+i+1 <= 16:
                agent_search[0] = agent_position[0]+i+1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[1]-i-1 >= 0:
                agent_search[1] = agent_position[1]-i-1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[1]+i+1 <= 16:
                agent_search[1] = agent_position[1]+i+1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True


        # check if a stone wall is between the agent an the bomb
        if is_in_bomb_range:
            if agent_position[0] == list(bomb[0])[0]:
                if agent_position[1] < list(bomb[0])[1]:
                    for i in range(agent_position[1], list(bomb[0])[1]):
                        if game_state['field'][agent_position[0]][i] == -1:
                            is_in_bomb_range = False
                else:
                    for i in range(list(bomb[0])[1], agent_position[1]):
                        if game_state['field'][agent_position[0]][i] == -1:
                            is_in_bomb_range = False
            elif agent_position[1] == list(bomb[0])[1]:
                if agent_position[0] < list(bomb[0])[0]:
                    for i in range(agent_position[0], list(bomb[0])[0]):
                        if game_state['field'][i][agent_position[1]] == -1:
                            is_in_bomb_range = False
                else:
                    for i in range(list(bomb[0])[0], agent_position[0]):
                        if game_state['field'][i][agent_position[1]] == -1:
                            is_in_bomb_range = False

    return is_in_bomb_range

def optimize_model(self):
    if len(self.memory) <= BATCH_SIZE:
        return
    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    #if not any(map(lambda x: x is None, batch.state)):
    state_batch = torch.cat(batch.state)

    reward_batch = torch.cat(batch.reward)

    action_list = []
    for idx, i in enumerate(batch.action):
        action_list.append(torch.tensor([[ACTIONS[batch.action[idx]]]], device=device))
    
    action_batch = torch.cat(action_list)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #self.loss_history.append(loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



