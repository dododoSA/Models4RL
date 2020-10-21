from models4rl.agents.base_agent import BaseAgent
from models4rl.replay_buffers.data_memory.data_memory import DataMemory
import torch
import torch.nn as nn
import torch.nn.functional as F

from models4rl.utils.network_utils import make_linear_network

state_num = 4
action_num = 2
hidden_size = 64


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        sizes = [state_num] + [hidden_size] * 2 + [1]
        self.fc = make_linear_network(sizes, nn.ELU, nn.ELU)

    def forward(self, x):
        return self.fc(x)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [state_num] + [hidden_size] + [action_num]
        self.fc = make_linear_network(sizes, nn.ELU, nn.Softmax)

    def forward(self, x):
        return self.fc(x)


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = PolicyNetwork()
        self.q = QNetwork()


import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(BaseAgent):
    def __init__(self, action_space, gamma=0.99):
        super().__init__(action_space, None, gamma)
        self.ac_network = ActorCriticNetwork()
        self.criterion = F.smooth_l1_loss

        self.p_optimizer = optim.Adam(self.ac_network.p.parameters(), lr=0.0005) # optimizer
        self.q_optimizer = optim.Adam(self.ac_network.q.parameters())

        self.trans_memory = DataMemory() # トランジションを保存するメモリ、ExperienceMemoryは経験再生用なのでこちらを使用

        self.state = None
        self.action = None
        self.episode = 1
        self.step = 1


    def act_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()

        self._append_memory(
            self.state,
            self.action,
            next_state,
            reward
        )
        
        self.state = next_state
        self.action = self._choice_greedy_action(next_state)

        self.step += 1
        
        return self.action
       


    def stop_episode_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()

        self._append_memory(
            self.state,
            self.action,
            next_state,
            reward
        )

        self.update()

        self.trans_memory.reset()
        self.state = None
        self.action = None

        self.episode += 1


    def update(self):
        R = 0
        actor_loss = 0
        critic_loss = 0
        for trans in self.trans_memory.get_latest_memories():
            r = trans["reward"]
            s = trans["state"]
            v = self.ac_network.q(s)
            action_prob = self.ac_network.p(s)[trans["action"]]

            R = r + self.gamma * R
            advantage = R - v
            actor_loss -= torch.log(action_prob) * advantage
            critic_loss += self.criterion(v, torch.tensor(R))

        actor_loss = actor_loss/len(self.trans_memory)
        critic_loss = critic_loss/len(self.trans_memory)
        
        self.p_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.p_optimizer.step()
        
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()



    def act_greedily(self, observation):
        next_state = torch.tensor(observation).float()
        return self._choice_greedy_action(next_state)


    def _choice_greedy_action(self, observation):
        """
        この関数まじでどうしよう
        """
        action_prob = self.ac_network.p(observation)
        # action = torch.argmax(action_prob.squeeze().data).item()
        action = Categorical(action_prob).sample().item()
        return action


    def _append_memory(self, state, action, next_state, reward):
        if state is None or action is None:
            return

        transition = {
            "state": state,
            "action": torch.LongTensor([[action]]), 
            "next_state": next_state,
            "reward": reward
        }

        self.trans_memory.append(transition)