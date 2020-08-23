from abc import ABCMeta, abstractmethod
from typing import Callable

class BaseExplorer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]) -> int:
        return 0

    @abstractmethod
    def end_episode(self):
        pass