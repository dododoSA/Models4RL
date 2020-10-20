from models4rl.agents.base_agent import BaseAgent
from models4rl.replay_buffers.data_memory.data_memory import DataMemory

class ActorCritic(BaseAgent):
    def __init__(self, ac_network, action_space, optimizer, gamma=0.99):
        super().__init__(action_space, None, gamma)
        self.optimizer = optimizer
        self.ac_network = ac_network
        self.data_memory = DataMemory() # トランジションを保存するメモリ、ExperienceMemoryは経験再生用なのでこちらを使用

        self.state = None
        self.action = None
        self.episode = 1
        self.step = 1


    def act_and_train(self):
        pass


    def stop_episode_and_train(self, observation, reward):
        pass