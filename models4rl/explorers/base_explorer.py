from abc import ABCMeta, abstractmethod
from typing import Callable

class BaseExplorer(metaclass=ABCMeta):
    def __init__(self):
        pass


    @abstractmethod
    def explore(self, random_action_func:Callable[[], int], greedy_action_func:Callable[[], int]) -> int:
        raise NotImplementedError()


    @abstractmethod
    def end_episode(self):
        raise NotImplementedError()