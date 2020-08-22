import numpy as np
from typing import Callable
from models.Explorer.Explorer import Explorer

class EpisodeLinearDecay(Explorer):
    def __init__(self, decay_episodes:int, start_epsilon:float=1.0, end_epsilon:float=0.0):
        assert start_epsilon >= end_epsilon, "'start_epsilon' must be equal to or greater than 'end_epsilon'"

        super(EpisodeLinearDecay, self).__init__()
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_episodes = decay_episodes
        self.episode = 1

    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]):
        if np.random.uniform(0, 1) <= self.epsilon:
            action = random_action_func()
        else:
            action = greedy_action_func()

        return action

    def end_episode(self):
        print(self.epsilon)
        self.episode += 1

        if self.episode < self.decay_episodes:
            self.epsilon = self.start_epsilon - (self.start_epsilon - self.end_epsilon) / self.decay_episodes * self.episode
        else:
            self.epsilon = self.end_epsilon
            