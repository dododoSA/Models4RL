import numpy as np
from typing import Callable
from models.explorers.base_explorer import BaseExplorer

class ConstantEpsilon(BaseExplorer):
    """
    イプシロンが定数値のε-greedy法
    ε = const

    Examples:
        explorer = ConstantEpsilon(epsilon=0.2)
        agent = Agent(explorer=explorer, args...)
    """
    def __init__(self, epsilon:float=0.1):
        """
        コンストラクタ

        Args:
            epsilon (float): 初期値
        """

        assert epsilon >= 0 and epsilon <= 1, "epsilon must be in [0, 1]"

        super(ConstantEpsilon, self).__init__()
        self.epsilon = epsilon

    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]) -> int:
        if np.random.uniform(0, 1) <= self.epsilon:
            action = random_action_func()
        else:
            action = greedy_action_func()

        return action

    def end_episode(self):
        pass