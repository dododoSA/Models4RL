import numpy as np
from typing import Callable
from models.explorers.base_explorer import BaseExplorer

class StepLinearDecay(BaseExplorer):
    """
    一つのエピソード内でステップ数が進むごとに、εが直線的に減少していくε-greedy法

    epsilon = - a * step + start_epsilon (epsilon in [end_epsilon, start_epsilon])

    Examples:
        explorer = EpisodeLinearDecay(decay_steps=100, start_epsilon=0.7, end_epsilon=0.01)
        agent = Agent(explorer=explorer, args...)
    """

    def __init__(self, decay_steps:int, start_epsilon:float=1.0, end_epsilon:float=0.0):
        """
        コンストラクタ

        Args:
            decay_steps (int): start_epsilonからend_epsilonに至るまでにかかるステップ数
            start_episilon (float): εの初期値
            end_epsilon (float): εが減少する最小値
        """

        assert start_epsilon >= end_epsilon, "'start_epsilon' must be equal to or greater than 'end_epsilon'"
        super(StepLinearDecay, self).__init__()
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.step = 1

    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]):
        if np.random.uniform(0, 1) <= self.epsilon:
            action = random_action_func()
        else:
            action = greedy_action_func()

        self._update_epsilon()

        return action

    def end_episode(self):
        self.step = 1
        self.epsilon = self.start_epsilon

    def _update_epsilon(self):
        self.step += 1

        if self.step < self.decay_steps:
            # 毎回直接もとめずに　(self.start_epsilon - self.end_epsilon) / self.decay_steps　を引くだけでもいいかもしれない
            self.epsilon = self.start_epsilon - (self.start_epsilon - self.end_epsilon) / self.decay_steps * self.step
        else:
            self.epsilon = self.end_epsilon
            