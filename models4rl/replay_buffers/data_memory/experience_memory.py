from models4rl.replay_buffers.data_memory.data_memory import DataMemory
import torch
import random

class ExperienceMemory(DataMemory):
    def __init__(self, capacity):
        super().__init__(capacity)


    def reshape_experience(self, experiences):
        states = torch.stack([experience["state"] for experience in experiences])
        next_states = torch.stack([experience["next_state"] for experience in experiences])
        actions = torch.cat([experience["action"] for experience in experiences])
        rewards = torch.tensor([experience["reward"] for experience in experiences])

        return { 'state': states, 'next_state': next_states, 'action': actions, 'reward': rewards }

    def make_batch(self, batch_size):
        experiences = random.sample(self.memory, batch_size)

        return self.reshape_experience(experiences)

