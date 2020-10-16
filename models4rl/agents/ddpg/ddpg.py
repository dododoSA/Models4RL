from models4rl.agents.base_agent import BaseAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


obs_num = 4 # 仮
node_num = 128 # 仮

class PolicyNetwork(nn.Module):
    def __init__():
        super().__init__()
        self.fc1 = nn.Linear(obs_num, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc4 = nn.Linear(node_num, 1)

        self.act_range = 1

    def forward(self, obs):
        x1 = F.tanh(self.fc1(obs))
        x2 = F.tanh(self.fc2(x1))
        x3 = F.tanh(self.fc3(x2))
        y = F.tanh(self.fc4(x3))
        return self.act_range * y


class Q_Network(nn.Module):
    def __init__():
        super().__init__()
        self.fc1 = nn.Linear(obs_num + 1, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc4 = nn.Linear(node_num, 1)

    def forward(self, obs, act):
        x1 = F.relu(self.fc1(torch.cat([obs, act])))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        q = F.relu(self.fc4(x3))
        return q.squeeze(1, -1)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = PolicyNetwork()
        self.q = Q_Network()


    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()



class DDPG(BaseAgent):
    def __init__(
        self,
        action_space,
        actor_critic,
        p_optimizer,
        q_optimizer,
        explorer,
        replay_buffer,
        batch_size=32,
        gamma=0.99
    ):
        super().__init__(action_space, explorer, gamma=gamma)

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.state = None
        self.action = None


    def act_and_train(self, observation, reward):
        pass

    def act_greedily(self, observation):
        pass

    def _choice_greedy_action(self, observation):
        pass

    def stop_episode_and_train(self, observation, reward):
        pass

