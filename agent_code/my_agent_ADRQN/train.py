import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import numpy as np

import events as e
import copy
from .model import Model
from .utils import state_to_features, save_rewards_to_file, save_num_steps_to_file
from .replay_memory import ReplayMemory, ReplayMemorySameEpisode
import torch
import torch.optim as optim
import torch.nn.functional as F
from .shared import device, EPS_START, EPS_END, EPS_DECAY
import agent_code.my_agent_ADRQN.global_model_variables


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
GAMMA = 0.9
TARGET_UPDATE = 20
#print(f'target update: {TARGET_UPDATE}')
NUM_EPISODES = 2000
LEARNING_RATE = 0.0001

USE_EPISODE_MEMORY = False
SEQ_LEN = 12

target_net = Model().to(device)
policy_net = Model().to(device)

if len(os.listdir("./agent_code/my_agent_ADRQN/saved_models/")) != 0:
    print('loading existing model...')
    policy_net.load_state_dict(torch.load("./agent_code/my_agent_ADRQN/saved_models/krasses_model.pt", map_location=torch.device('cpu')))
    #policy_net.load_state_dict(torch.load("./agent_code/my_agent/saved_models/krasses_model.pt"))
    policy_net.eval().train()

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
    self.current_episode_num = 1
    self.total_reward = 0

    #self.optimizer = optim.RMSprop(policy_net.parameters())
    if USE_EPISODE_MEMORY:
        self.memory = ReplayMemorySameEpisode(400000, SEQ_LEN)
        self.memory.start_episode()
    else:
        self.memory = ReplayMemory(400000, SEQ_LEN)

    self.total_reward_history = []
    self.total_steps_done_history = []
    #self.loss_history = []
    self.positions = []
    self.last_action = 4

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
    #print(f'steps_done in game event: {self.steps_done}')
    min_coin_distance = calc_coin_distance(new_game_state)

    if calc_position_change(self, new_game_state):
        events.append('REPEATS_STEPS')
    if calc_is_new_position(self, new_game_state):
        events.append('NEW_POSITION_EXPLORED')

    # set reward if bomb is dropped near crates
    if 'BOMB_DROPPED' in events:
        self.n_destroyed_crates = crates_destroyed(self, new_game_state)

    is_in_bomb_range, min_bomb_distance = in_bomb_range(self, new_game_state)
    # set variable is_in_bomb_range
    if is_in_bomb_range:
        events.append('IN_BOMB_RANGE')

    reward = reward_from_events(self, events, min_coin_distance, min_bomb_distance, new_game_state)
    
    self.total_reward += reward
    reward = torch.tensor([reward], device=device)
    if  state_to_features(old_game_state) != None:
        #self.memory.push(state_to_features(old_game_state).to(device), self_action, state_to_features(new_game_state).to(device), reward)
        self.memory.write_tuple((torch.tensor(self.last_action).to(device),
                                 state_to_features(old_game_state).to(device),
                                 torch.tensor(ACTIONS[self_action]).to(device),
                                 torch.tensor(reward).to(device),
                                 state_to_features(new_game_state).to(device),
                                 torch.tensor(False).to(device)))
    if self_action != None:
        self.last_action = ACTIONS[self_action]
    self.positions.append(new_game_state['self'][3])

    self.old_game_state = old_game_state
    self.last_old_game_state = new_game_state
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
    min_coin_distance = calc_coin_distance(last_game_state)

    # if calc_position_change(self, new_game_state):
    #    events.append('REPEATS_STEPS')
    if calc_is_new_position(self, last_game_state):
        events.append('NEW_POSITION_EXPLORED')

    # set reward if bomb is dropped near crates
    if 'BOMB_DROPPED' in events:
        self.n_destroyed_crates = crates_destroyed(self, last_game_state)

    is_in_bomb_range, min_bomb_distance = in_bomb_range(self, last_game_state)
    # set variable is_in_bomb_range
    if is_in_bomb_range:
        events.append('IN_BOMB_RANGE')

    reward = reward_from_events(self, events, min_coin_distance, min_bomb_distance, last_game_state)

    self.total_reward += reward

    #self.memory.push(state_to_features(self.old_game_state), last_action, state_to_features(last_game_state), self.total_reward_tensor)
    if  state_to_features(self.last_old_game_state) != None:
        #self.memory.push(state_to_features(old_game_state).to(device), self_action, state_to_features(new_game_state).to(device), reward)
        self.memory.write_tuple((torch.tensor(self.last_action).to(device),
                                 state_to_features(self.last_old_game_state).to(device),
                                 torch.tensor(ACTIONS[last_action]).to(device),
                                 torch.tensor(reward).to(device),
                                 state_to_features(last_game_state).to(device),
                                 torch.tensor(True).to(device)))
    if last_action != None:
        self.last_action = ACTIONS[last_action]
    optimize_model(self)


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    if self.current_episode_num % TARGET_UPDATE == 0:
        print(f"saving model...{self.current_episode_num}")
        torch.save(target_net.state_dict(), "./saved_models/krasses_model.pt")
        save_rewards_to_file(self.total_reward_history)
        save_num_steps_to_file(self.total_steps_done_history)
        print('update target_net')
        target_net.load_state_dict(policy_net.state_dict())

    print(f'finished episode {self.current_episode_num}')

    
    self.total_reward_history.append(self.total_reward)
    self.total_steps_done_history.append(last_game_state['step'])
    if self.current_episode_num == NUM_EPISODES:
        print(f"saving model...{self.current_episode_num}")
        torch.save(target_net.state_dict(), "./saved_models/krasses_model.pt")
        save_rewards_to_file(self.total_reward_history)
        #save_loss_to_file(self.loss_history)
    #print(f'TOTAL REWARD: {self.total_reward}')
    #print(f'steps done: {self.steps_done}')




    self.total_reward = 0
    self.current_episode_num += 1

    #print(f'Positions: {len(self.positions)}')
    self.positions = []
    self.last_action = 4
    agent_code.my_agent_ADRQN.global_model_variables.last_action = 4
    steps = last_game_state['step']
    print(f'Number of steps: {steps}')
    if USE_EPISODE_MEMORY:
        self.memory.start_episode()

def reward_from_events(self, events: List[str], distance_coin, distance_bomb, game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.

    events: list of events
    distance_coin: distance of agent to the nearest coin
    """
    #print(f'Events: {events}')
    game_rewards = {
        e.COIN_COLLECTED: 6000,
        e.KILLED_OPPONENT: 0,
        e.INVALID_ACTION: -1000,
        e.MOVED_DOWN: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.WAITED: -1000,
        e.BOMB_DROPPED: 15000,
        e.IN_BOMB_RANGE: -30,
        e.KILLED_SELF: -2000,
        #e.CRATE_DESTROYED: 1000
        #e.REPEATS_STEPS: -5000
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    #print(f'Reward from events: {reward_sum}')
    #set reward for the coin distance
    if len(game_state['coins']) != 0:
        reward_sum += int(1000 - distance_coin * 1000)

    #set reward for the bomb distance
    for event in events:
        if event == e.IN_BOMB_RANGE:
            # ca. distance for one step: 0.046
            # ca. distance for two steps: 0.093
            reward_sum += int((distance_bomb*27.73*1000-4000)/4)
    #print(f'Reward bomb distance: {reward_sum}')
    # no bombs in corners
    corners = [(1, 1), (1, 15), (15, 15), (15, 1)]
    if 'BOMB_DROPPED' in events:
        for corner in corners:
            if corner == game_state['self'][3]:
                reward_sum -= 8000

    #print(f'Reward corners: {reward_sum}')
    # set reward for destroyed crates

    reward_sum += self.n_destroyed_crates * 2000
    self.n_destroyed_crates = 0
    #print(f'Reward destroyed crates: {reward_sum}')
    # normalize the calculated reward
    max_reward = -500
    min_reward = -2100
    for key, value in game_rewards.items():
        if value > 0:
            max_reward += value
        else:
            min_reward += value
    #print(f'minreward: {min_reward}')
    #print(f'maxreward: {max_reward}')

    reward_sum = (reward_sum - min_reward) / (max_reward-min_reward) * 3
    #print(f'Reward: {reward_sum}')
    return reward_sum

def calc_coin_distance(game_state: dict):
    # maximum distance an agent could have to a coin
    max_distance = 21.5
    # minimum distance an agent could (theoretically) have to a coin
    min_distance = 0

    """calculate distance of agent to next coin (a value between 0 and 1)"""
    min_coin_distance = float('inf')
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

def crates_destroyed(self, game_state: dict):
    """

    returns the number of crates, which will be destroyed by a bomb

    iterate through every line in which the bomb will be destroyed.
    """

    bomb_position_x = game_state['self'][3][0]
    bomb_position_y = game_state['self'][3][1]
    n_crates = 0

    for i in range(3):
        if bomb_position_x - i - 1 >= 0:
            if game_state['field'][bomb_position_x - i - 1][bomb_position_y] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x - i - 1][bomb_position_y] == -1:
                break

    for i in range(3):
        if bomb_position_x + i + 1 <= 16:
            if game_state['field'][bomb_position_x + i + 1][bomb_position_y] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x + i + 1][bomb_position_y] == -1:
                break

    for i in range(3):
        if bomb_position_y - i - 1 >= 0:
            if game_state['field'][bomb_position_x][bomb_position_y - i - 1] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x][bomb_position_y - i - 1] == -1:
                break

    for i in range(3):
        if bomb_position_y + i + 1 <= 16:
            if game_state['field'][bomb_position_x][bomb_position_y + i + 1] == 1:
                n_crates += 1
            elif game_state['field'][bomb_position_x][bomb_position_y + i + 1] == -1:
                break

    return n_crates


def in_bomb_range(self, game_state: dict):
    """

    returns true, if the agent is in the bomb explosion radius

    """
    is_in_bomb_range = False
    min_bomb_distance = 0
    agent_position = game_state['self'][3]
    agent_position = list(agent_position)

    for bomb in game_state['bombs']:
        if agent_position == list(bomb[0]):
            is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[0] - i - 1 >= 0:
                agent_search[0] = agent_position[0] - i - 1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[0] + i + 1 <= 16:
                agent_search[0] = agent_position[0] + i + 1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[1] - i - 1 >= 0:
                agent_search[1] = agent_position[1] - i - 1
                if agent_search == list(bomb[0]):
                    is_in_bomb_range = True

        for i in range(3):
            agent_search = copy.copy(agent_position)
            if agent_position[1] + i + 1 <= 16:
                agent_search[1] = agent_position[1] + i + 1
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

    if is_in_bomb_range:
        min_bomb_distance = calc_bomb_distance(game_state)

    return is_in_bomb_range, min_bomb_distance

def calc_bomb_distance(game_state: dict):
    # maximum distance an agent could have to a coin
    max_distance = 21.5
    # minimum distance an agent could (theoretically) have to a coin
    min_distance = 0

    """calculate distance of agent to next coin (a value between 0 and 1)"""
    min_bomb_distance = float('inf')
    agent_position = np.asarray(game_state['self'][3])
    for bomb in game_state['bombs']:
        dist = np.linalg.norm(agent_position - np.asarray(bomb[0]))
        if min_bomb_distance > dist:
            min_bomb_distance = dist

    # normalize to a value between 0 and 1
    return (min_bomb_distance - min_distance) / (max_distance - min_distance)

def optimize_model_depr(self):
    if len(self.memory) <= BATCH_SIZE:
        return
    if USE_EPISODE_MEMORY:
        transitions = self.memory.sample(BATCH_SIZE, SEQ_LEN)
    else:
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

def optimize_model(self):
    if len(self.memory) <= BATCH_SIZE:
        return

    if USE_EPISODE_MEMORY:
        last_actions, last_observations, actions, rewards, observations, dones = self.memory.sample(BATCH_SIZE, SEQ_LEN)
    else:
        last_actions, last_observations, actions, rewards, observations, dones = self.memory.sample(BATCH_SIZE)

    #print(f'Rewards: {rewards}')
    # Pass the sequence of last observations and actions through the network
    q_values, _ = policy_net.forward(last_observations, F.one_hot(last_actions, 6).float())
    # Get the q_values for the executed actions in the respective observations
    q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
    # Query the target network for Q value predictions
    predicted_q_values, _ = target_net.forward(observations, F.one_hot(actions, 6).float())
    # Compute Q update target
    target_values = rewards + (GAMMA * (1 - dones.float()) * torch.max(predicted_q_values, dim=-1)[0])
    # Updating network parameters
    optimizer.zero_grad()
    loss = torch.nn.MSELoss()(q_values, target_values.detach())
    loss.backward()
    optimizer.step()


