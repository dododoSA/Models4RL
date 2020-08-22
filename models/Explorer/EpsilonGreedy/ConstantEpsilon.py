import numpy as np
from typing import Callable
from models.Explorer.Explorer import Explorer

class ConstantEpsilon(Explorer):
    """

    """
    def __init__(self, epsilon=0.1):
        super(ConstantEpsilonGreedy, self).__init__()
        self.epsilon = epsilon

    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]):
        if np.random.uniform(0, 1) <= self.end_epsilon:
            action = random_action_func()
        else:
            action = greedy_action_func()

        return action

    def end_episode(self):
        super().end_episode()