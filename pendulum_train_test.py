from models4rl.agents.ddpg.ddpg import DDPG
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


env = gym.make('Pendulum-v0')
max_steps = 200
episode_num = 1000

# explorer = EpisodeLinearDecay(700, 0.8, 0)
#explorer = StepLinearDecay(max_steps - 100, 0.1, 0)
#explorer = EpisodeExpDecay(a=0.99)
explorer = ConstantEpsilon(1)

from models4rl.agents.ddpg.ddpg import ActorCritic

actor_critic = ActorCritic()
p_optimizer = optim.Adam(actor_critic.pi.parameters(), lr=0.01)
q_optimizer = optim.Adam(actor_critic.q.parameters(), lr=0.01)
#criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(10000)

agent = DDPG(
    env.action_space,
    actor_critic,
    p_optimizer,
    q_optimizer,
    explorer,
    replay_buffer
)


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

        action = agent.act_greedily(obs)

        obs, reward, done, _ = env.step(action)

        episode_reward += reward
        t += 1
        
    print("episode: ", i, " episode reward: ", episode_reward)

env.close()