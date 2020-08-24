import random

class ReplayBuffer():
    """
    ここのパクリ
    https://book.mynavi.jp/manatee/detail/id=89831
    キューっぽくcapacityに達したらデータを削除するようにする方式も考えたけどappendはメモリ確保に時間がかかるっぽいからこれが速そう
    めちゃくちゃ余裕があるならmemoryのデータ構造を自作するのはあり
    優先度付き経験再生など他の経験再生のために抽象化する可能性あり
    """
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def append(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.memory[self.index] = {
                "state": state,
                "action": action, 
                "next_state": next_state,
                "reward": reward,
                "done": done
            }

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.memory)