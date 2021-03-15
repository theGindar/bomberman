import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.n_actions = 6
        self.embedding_size = 12
        self.embedder = nn.Linear(self.n_actions, self.embedding_size)
        #self.obs_layer = nn.Linear(state_size, 16)
        #self.obs_layer2 = nn.Linear(16, 32)
        self.obs_conv1 = nn.Conv3d(6, 32, (1,2,2), 1)
        self.obs_conv2 = nn.Conv3d(32, 64, (1,2,2), (1,2,2))
        self.obs_conv3 = nn.Conv3d(64, 64, (1,2,2), 1)
        self.obs_fc4 = nn.Linear(7 * 7 * 64, 512)
        #self.obs_fc5 = nn.Linear(512, 32)
        self.lstm = nn.LSTM(input_size=512 + self.embedding_size, hidden_size=128, batch_first=True)
        self.out_layer = nn.Linear(128, self.n_actions)

    def forward(self, observation, action, hidden=None):
        # Takes observations with shape (batch_size, seq_len, state_size)
        # Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        action_embedded = self.embedder(action)
        observation = F.relu(self.obs_conv1(observation))
        observation = F.relu(self.obs_conv2(observation))
        observation = F.relu(self.obs_conv3(observation))

        observation = observation.permute(0,2,1,3,4)
        observation = observation.contiguous()
        observation = observation.view(observation.size()[0], observation.size()[1], -1)
        observation = F.relu(self.obs_fc4(observation))
        lstm_input = torch.cat([observation, action_embedded], dim=-1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)

        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out

    def act(self, observation, last_action, hidden=None):
        q_values, hidden_out = self.forward(observation, last_action, hidden)
        action = torch.argmax(q_values).item()
        return action, hidden_out