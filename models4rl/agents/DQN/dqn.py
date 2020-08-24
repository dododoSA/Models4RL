import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models4rl.replay_buffers.replay_buffer import ReplayBuffer

obs_num = 4 # 仮
node_num = 128 # 仮
act_num = 2 # 仮

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




class DQN():
    def __init__():
        self.q_network = Q_Network()
        self.replay_memory = ReplayBuffer(10000)

        self.optimizer = optim.Adam(self.q_netwark.parameters(), lr=0.0001)

    def act_and_train():
        pass