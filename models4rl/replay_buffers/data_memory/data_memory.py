class DataMemory():
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def append(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = data

        self.index = (self.index + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    
    def get_memory(self):
        return self.memory


    def get_capacity(self):
        return self.get_capacity