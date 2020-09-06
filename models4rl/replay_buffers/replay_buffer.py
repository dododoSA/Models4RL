import random
import numpy as np
import torch

class ReplayBuffer():
    """
    ここのほぼパクリ
    https://book.mynavi.jp/manatee/detail/id=89831
    キューっぽくcapacityに達したらデータを削除するようにする方式も考えたけどappendはメモリ確保に時間がかかるっぽいからこれが速そう
    めちゃくちゃ余裕があるならmemoryのデータ構造を自作するのはあり
    優先度付き経験再生など他の経験再生のために抽象化する可能性あり
    """
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def append(self, state, action, next_state, reward):
        if state is None or action is None:
            return

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.index] = {
                "state": state,
                "action": torch.LongTensor([[action]]), 
                "next_state": next_state,
                "reward": reward
            }

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        states = torch.stack([experience["state"] for experience in experiences])
        next_states = torch.stack([experience["next_state"] for experience in experiences])
        actions = torch.cat([experience["action"] for experience in experiences])
        rewards = torch.tensor([experience["reward"] for experience in experiences])

        return { 'state': states, 'next_state': next_states, 'action': actions, 'reward': rewards }

    def __len__(self):
        return len(self.buffer)