from models4rl.replay_buffers.data_memory.data_memory import DataMemory

import numpy as np

class TDerrorMemory(DataMemory):
    def __init__(self, capacity, epsilon):
        super().__init__(capacity)
        self.epsilon = epsilon

    def get_prioritized_indexes(self, batch_size):
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += self.epsilon * len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += abs(self.memory[idx]) + self.epsilon
                idx += 1

            idx %= len(self.memory)

            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors