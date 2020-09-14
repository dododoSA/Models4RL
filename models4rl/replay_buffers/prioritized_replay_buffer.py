import random
import numpy as np
import torch

from models4rl.replay_buffers.base_buffer import BaseBuffer
from models4rl.replay_buffers.data_memory.experience_memory import ExperienceMemory
from models4rl.replay_buffers.data_memory.tderror_memory import TDerrorMemory

class PrioritizedReplayBuffer(BaseBuffer):
    def __init__(self, capacity, start_size=128, epsilon=0.0001):
        self.experience_memory = ExperienceMemory(capacity)
        self.tderror_memory = TDerrorMemory(capacity, epsilon)

    def append_and_update(self, state, action, next_state, reward, **keys):
        if state is None or action is None:
            return

        transition = {
            "state": state,
            "action": torch.LongTensor([[action]]), 
            "next_state": next_state,
            "reward": reward
        }

        self.experience_memory.append(transition)

        # バッチサイズ以下の時は無駄な処理な気もするけれど、バッチサイズはせいぜい数百なのでスルー
        self._update_td_error_memory(keys['q_network'], keys['target_network'], keys['gamma'])


    def get_batch(self, batch_size):
        #if len(self.experience_memory) < self.experience_memory.capacity:
        #    experiences = self.experience_memory.make_batch(batch_size)
        #    return self.experience_memory.make_batch(batch_size)

        indexes = self.tderror_memory.get_prioritized_indexes(batch_size)
        tmp = self.experience_memory.get_memory()
        experiences = [tmp[i] for i in indexes]

        return self.experience_memory.reshape_experience(experiences)


    def _update_td_error_memory(self, q_network, target_network, gamma):
        all_experiences = self.experience_memory.get_memory()
        state_batch, next_state_batch, action_batch, reward_batch = self.experience_memory.reshape_experience(all_experiences).values()


        q_network.eval()

        q_values = q_network(state_batch).gather(1, action_batch)
        
        
        target_network.eval()

        next_q_values = torch.zeros_like(q_values, dtype=float)
        next_q_values = target_network(next_state_batch).max(1)[0].detach()


        td_errors = (reward_batch + gamma * next_q_values) - q_values.squeeze()

        self.tderror_memory.update_td_error(td_errors.detach().numpy().tolist())

    def __len__(self):
        return len(self.experience_memory)