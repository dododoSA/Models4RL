from models4rl.explorers.base_explorer import BaseExplorer

from abc import ABCMeta, abstractmethod
import numpy as np
from gym.spaces.discrete import Discrete

class BaseAgent(metaclass=ABCMeta):
    def __init__(self, action_space:Discrete, explorer:BaseExplorer, gamma:float=0.99):
        self.action_space = action_space
        self.explorer = explorer
        self.gamma = gamma

    @abstractmethod
    def act_and_train(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _choice_greedy_action(self, observation:np.ndarray) -> int:
        raise NotImplementedError()

    @abstractmethod
    def act_greedily(self, observation:np.ndarray) -> int:
        # _choice_greedy_actionを呼び出すだけだから抽象関数にしなくてもいいかもしれない
        raise NotImplementedError()

    @abstractmethod
    def stop_episode_and_train(self, observation:np.ndarray, reward:float) -> None:
        raise NotImplementedError()