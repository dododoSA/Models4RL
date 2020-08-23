import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

obs_num = 4 # 仮
node_num = 128 # 仮
act_num = 2

class Q_Network(nn.Module):
    def __init__(self):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(obs_num, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc4 = nn.Linear(node_num, act_num)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = F.relu(self.fc4(h))
        return y

class ReplayMemory:
    def __init__(self):
        self.memory = []

    def sample():
        pass


class DQN():
    def __init__():
        pass

