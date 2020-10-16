import random
import numpy as np
import torch

from models4rl.replay_buffers.base_buffer import BaseBuffer
from models4rl.replay_buffers.data_memory.experience_memory import ExperienceMemory


class ReplayBuffer(BaseBuffer):
    """
    ここのほぼパクリ
    https://book.mynavi.jp/manatee/detail/id=89831
    キューっぽくcapacityに達したらデータを削除するようにする方式も考えたけどappendはメモリ確保に時間がかかるっぽいからこれが速そう
    めちゃくちゃ余裕があるならmemoryのデータ構造を自作するのはあり
    """
    def __init__(self, capacity):
        self.experience_memory = ExperienceMemory(capacity)

    def append_and_update(self, state, action, next_state, reward, **keys={}):
        if state is None or action is None:
            return

        transition = {
            "state": state,
            "action": torch.LongTensor([[action]]), 
            "next_state": next_state,
            "reward": reward
        }

        self.experience_memory.append(transition)

    def get_batch(self, batch_size):
        return self.experience_memory.make_batch(batch_size)

    def __len__(self):
        return len(self.experience_memory)
