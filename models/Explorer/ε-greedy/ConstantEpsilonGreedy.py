import numpy as np
from models.Explorer import Explorer

class ConstantEpsilonGreedy(Explorer):
    def __init__(self, epsilon=0.1):
        super(ConstantEpsilonGreedy, self).__init__()
        self.epsilon = epsilon

    def explore(self, random_action_func:function, greedy_action_func:function):
        if self.epsilon <= np.random.uniform(0, 1):
            action = greedy_action_func()
        else:
            action = random_action_func()

        return action