import numpy as np
from typing import Callable
from models4rl.explorers.base_explorer import BaseExplorer

class EpisodeExpDecay(BaseExplorer):
    """
    エピソード数が増えるごとに、指数関数的にεが減少していくε-greedy法

    epsilon = start_epsilon * a ** (episode - 1)

    Examples:
        explorer = EpisodeLinearDecay(start_epsilon=0.7, a=0.999)
        agent = Agent(explorer=explorer, args...)
    """
    def __init__(self, start_epsilon:float=1.0, a:float=0.99):
        """
        コンストラクタ

        Args:
            start_epsilon: εの初期値
            a: 毎エピソードεに掛ける値
        """
        assert a <= 1, "a must not be greater than 1"

        super(EpisodeExpDecay, self).__init__()
        self.epsilon = start_epsilon
        self.a = a


    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]) -> int:
        if np.random.uniform(0, 1) <= self.epsilon:
            action = random_action_func()
        else:
            action = greedy_action_func()

        return action


    def end_episode(self) -> None:
        self.epsilon *= self.a
            