class DataMemory():
    def __init__(self, capacity:int=None):
        self.capacity = capacity
        self.reset_memory()

    def append(self, data):

        can_append = self.capacity is None or len(self.memory) < self.capacity
        if can_append:
            self.memory.append(None)

        self.memory[self.index] = data

        self.index +=  1
        if not self.capacity is None:
            self.index %= self.capacity


    def __len__(self):
        return len(self.memory)

    
    def get_memory(self):
        return self.memory


    def get_capacity(self):
        return self.get_capacity


    def reset(self):
        self.memory = []
        self.index = 0


    def get_latest_memories(self, num=None):
        """
        新しい順にメモリーを取り出す関数
        逆順になっていることに注意
        """
        if num is None:
            num = len(self.memory)

        reversed_memory = self.memory[::-1]
        return reversed_memory[0:num]