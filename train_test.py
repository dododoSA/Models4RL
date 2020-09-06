from models4rl.agents.q_learning.q_learning import Qlearning
from models4rl.agents.dqn.dqn import DQN
from models4rl.explorers.epsilon_greedy.constant_epsilon import ConstantEpsilon
from models4rl.explorers.epsilon_greedy.episode_linear_decay import EpisodeLinearDecay
from models4rl.explorers.epsilon_greedy.step_linear_decay import StepLinearDecay
from models4rl.explorers.epsilon_greedy.episode_exp_decay import EpisodeExpDecay
from models4rl.replay_buffers.replay_buffer import ReplayBuffer
import gym
import time
import numpy as np
import collections
import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK']='True'


env = gym.make('CartPole-v0')
max_steps = 200
episode_num = 1000

explorer = EpisodeLinearDecay(500, 0.7, 0)
#explorer = StepLinearDecay(max_steps - 100, 0.1, 0)
#explorer = EpisodeExpDecay(a=0.99)
#explorer = ConstantEpsilon(0)

# agent = Qlearning([9,2,8,2], env.observation_space, env.action_space,
# explorer, init_q_max=0.01)


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

q_network = Q_Network()
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(1000)
agent = DQN(env.action_space, q_network, optimizer, criterion, explorer, replay_buffer, target_update_episode_interval=15
            )

def compute_reward(reward, done):
    if done:
        if t < max_steps - 5:
            reward = -10
        else:
            reward = 1
    return reward

queue = []
ave_10_episodes_reward = []

for episode in range(episode_num):
    obs = env.reset()
    episode_reward = 0
    done = False
    reward = 0

    t = 0
    while t < max_steps and not done:
        action = agent.act_and_train(obs, reward)

        obs, reward, done, _ = env.step(action)

        reward = compute_reward(reward, done)

        episode_reward += reward
        t += 1
    
    queue.append(episode_reward)
    if len(queue) == 11:
        queue.pop(0)

    ave_10_episodes_reward.append(sum(queue) / len(queue))


    agent.stop_episode_and_train(obs, reward)
    print("episode: ", episode, " episode reward: ", episode_reward)

    
plt.plot(range(1 + episode), ave_10_episodes_reward)
plt.xlabel('episodes [1, 1500]')
plt.ylabel('reward')
plt.show()

for i in range(5):
    obs = env.reset()
    episode_reward = 0
    done = False
    reward = 0
    t = 0
    while not done:
        env.render()

        action = agent.act_greedy(obs)

        obs, reward, done, _ = env.step(action)

        reward = compute_reward(reward, done)

        episode_reward += reward
        t += 1
        
    print("episode: ", i, " episode reward: ", episode_reward)

env.close()