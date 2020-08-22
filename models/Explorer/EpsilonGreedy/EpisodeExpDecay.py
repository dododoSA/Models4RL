import numpy as np
from typing import Callable
from models.Explorer.Explorer import Explorer

class EpisodeExpDecay(Explorer):
    def __init__(self, start_epsilon:float=1.0, a:float=0.99):
        assert a <= 1, "a must not be greater than 1"

        super(EpisodeExpDecay, self).__init__()
        self.epsilon = start_epsilon
        self.a = a

    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]):
        if np.random.uniform(0, 1) <= self.epsilon:
            action = random_action_func()
        else:
            action = greedy_action_func()

        return action

    def end_episode(self):
        self.epsilon *= self.a
            