import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer1 = nn.Conv2d(6, 32, kernel_size=2, stride=1)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.dense_layer4 = nn.Linear(7 * 7 * 64, 512)
        self.dense_layer5 = nn.Linear(512, 6)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        x = F.relu(self.conv_layer3(x))
        x = F.relu(self.dense_layer4(x.view(x.size(0), -1)))
        return self.dense_layer5(x)