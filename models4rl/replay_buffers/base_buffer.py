from abc import ABCMeta, abstractmethod

class BaseBuffer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def append_and_update(self, state, action, next_state, reward, **keys):
        raise NotImplementedError()

    @abstractmethod
    def get_batch(self, batch_size):
        raise NotImplementedError()