from models4rl.agents.base_agent import BaseAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


obs_num = 4 # 仮
node_num = 128 # 仮

class PolicyNetwork(nn.Module):
    def __init__():
        super().__init__()
        self.fc1 = nn.Linear(obs_num, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc4 = nn.Linear(node_num, 1)

        self.act_range = 1

    def forward(self, obs):
        x1 = F.tanh(self.fc1(obs))
        x2 = F.tanh(self.fc2(x1))
        x3 = F.tanh(self.fc3(x2))
        y = F.tanh(self.fc4(x3))
        return self.act_range * y


class Q_Network(nn.Module):
    def __init__():
        super().__init__()
        self.fc1 = nn.Linear(obs_num + 1, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, node_num)
        self.fc4 = nn.Linear(node_num, 1)

    def forward(self, obs, act):
        x1 = F.relu(self.fc1(torch.cat([obs, act])))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        q = F.relu(self.fc4(x3))
        return q.squeeze(1, -1)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = PolicyNetwork()
        self.q = Q_Network()


    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()



class DDPG(BaseAgent):
    def __init__(
        self,
        action_space,
        actor_critic,
        p_optimizer,
        q_optimizer,
        explorer,
        replay_buffer,
        batch_size=32,
        gamma=0.99
    ):
        super().__init__(action_space, explorer, gamma=gamma)

        ac = actor_critic
        ac_target = deepcopy(ac)

        # ↓これって本当に必要なん？
        for p in ac_target.parameters():
            p.requires_grad = False

        self.p_optimizer = p_optimizer
        self.q_optimizer = q_optimizer

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.state = None
        self.action = None


    def act_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()
        
        self.replay_buffer.append_and_update(
            self.state,
            self.action,
            next_state,
            reward
        )
        self.replay() # imakoko

        if self.target_update_step_interval and self.step % self.target_update_step_interval == 0:
            self._update_target_network()

        self.state = next_state
        self.action = self.explorer.explore(self.action_space.sample, lambda: self._choice_greedy_action(next_state))

        self.step += 1
        return self.action


    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, next_state_batch, action_batch, reward_batch = self.replay_buffer.get_batch(self.batch_size).values()
        
        
        # 自分の過去の行動から現在の方策の価値を計算
        q_values = self.ac.q(state_batch, action_batch)

        with torch.no_grad():
            q_target = self.ac_target.q(next_state_batch, ac_target.pi(next_state_batch))
            backup = reward_batch + self.gamma * q_target

        # MSE Loss
        loss_q = ((q_values - backup)**2).mean()

        # 現在の方策を評価
        loss_pi = - self.ac.q(state_batch, ac.pi(state_batch)).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        q_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = False


        self.p_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()

        for p in ac.q.parameters():
            p.requires_grad = True

        # polyak averaging
        with torch.no_grad():
            for p, p_target in zip(ac.parameters(), ac_target.parameters()):
                p_target.data.mul_(0.995)
                p_target.data.add_((1 - 0.995) * p.data)



        #self.q_network.eval()

        #q_values = self.q_network(state_batch).gather(1, action_batch)
        

        #self.target_network.eval()

        #next_q_values = torch.zeros_like(q_values, dtype=float)
        #next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        #target_values = self.gamma * next_q_values + reward_batch


        #self.q_network.train()
        
        #loss = self.criterion(q_values, target_values.unsqueeze(1).float())
        #self.optimizer.zero_grad()
        #loss.backward()

        ##for p in self.q_network.parameters():
        ##    p.grad.data.clamp_(-1, 1)
        #self.optimizer.step()

    def act_greedily(self, observation):
        action = self.ac.act(torch.as_tensor(observation, dtype=torch.float32))
        return np.clip(a, -1, 1) # todo fix

    def _choice_greedy_action(self, observation):
        pass

    def stop_episode_and_train(self, observation, reward):
        pass

