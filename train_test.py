from models.Qlearning.Qlearning import Qlearning
import gym
import time
import numpy as np
import collections
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
max_steps = 200
episode_num = 1500

env.observation_space.high[1]=2
env.observation_space.low[1]=-2
env.observation_space.high[3]=2
env.observation_space.low[3]=-2

agent = Qlearning(8, env.observation_space, env.action_space, init_q_max=0.01)

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
    print("episode: " + str(episode) + " episode reward: " + str(episode_reward))

    
plt.plot(range(1 + episode), ave_10_episodes_reward)
plt.show()
plt.pause(0.001)

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
        
    print("episode: " + str(i) + " episode reward: " + str(episode_reward))