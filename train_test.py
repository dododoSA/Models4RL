from models.agents.q_learning.q_learning import Qlearning
from models.explorers.epsilon_greedy.constant_epsilon import ConstantEpsilon
from models.explorers.epsilon_greedy.episode_linear_decay import EpisodeLinearDecay
from models.explorers.epsilon_greedy.step_linear_decay import StepLinearDecay
from models.explorers.epsilon_greedy.episode_exp_decay import EpisodeExpDecay
import gym
import time
import numpy as np
import collections
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')
max_steps = 200
episode_num = 2000

explorer = EpisodeLinearDecay(1000, 0.01, 0)
#explorer = StepLinearDecay(max_steps - 100, 0.1, 0)
#explorer = EpisodeExpDecay(a=0.99)
#explorer = ConstantEpsilon(0)

agent = Qlearning([9,2,8,2], env.observation_space, env.action_space, explorer, init_q_max=0.01)

def compute_reward(reward, done):
    if done:
        if t < max_steps - 5:
            reward = -200
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

    ave_10_episodes_reward.append(sum(queue)/len(queue))


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