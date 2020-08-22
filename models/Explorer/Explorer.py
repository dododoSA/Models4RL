from abc import ABCMeta, abstractmethod

class Explorer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def explore(self, random_action_func:function, greedy_action_func:function) -> int:
        return 0
