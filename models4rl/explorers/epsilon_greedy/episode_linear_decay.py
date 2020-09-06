import numpy as np
from typing import Callable
from models4rl.explorers.base_explorer import BaseExplorer

class EpisodeLinearDecay(BaseExplorer):
    """
    エピソード数が増えるごとに、直線的にεが減少していくε-greedy法

    epsilon = - a * episode + start_epsilon (epsilon in [end_epsilon, start_epsilon])
    
    Examples:
        explorer = EpisodeLinearDecay(decay_episodes=100, start_epsilon=0.7, end_epsilon=0.01)
        agent = Agent(explorer=explorer, args...)
    """

    def __init__(self, decay_episodes:int, start_epsilon:float=1.0, end_epsilon:float=0.0):
        """
        コンストラクタ

        Args:
            decay_episodes (int): start_epsilonからend_epsilonに至るまでにかかるエピソード数
            start_episilon (float): εの初期値
            end_epsilon (float): εが減少する最小値

        """
        assert start_epsilon >= end_epsilon, "'start_epsilon' must be equal to or greater than 'end_epsilon'"

        assert start_epsilon >= 0 and start_epsilon <= 1, "epsilon must be in [0, 1]"
        assert end_epsilon >= 0 and end_epsilon <= 1, "epsilon must be in [0, 1]"

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
        self.episode += 1

        if self.episode < self.decay_episodes:
            # 毎回直接もとめずに　(self.start_epsilon - self.end_epsilon) / self.decay_episodes　を引くだけでもいいかもしれない
            self.epsilon = self.start_epsilon - (self.start_epsilon - self.end_epsilon) / self.decay_episodes * self.episode
        else:
            self.epsilon = self.end_epsilon
            