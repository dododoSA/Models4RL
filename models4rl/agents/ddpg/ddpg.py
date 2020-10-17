from models4rl.agents.base_agent import BaseAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

"""
優先度付き経験再生は未対応
"""


obs_num = 3 # 仮
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
    """
    
    """
    def __init__(
        self,
        action_space,
        actor_critic,
        p_optimizer,
        q_optimizer,
        explorer,
        replay_buffer,
        batch_size=32,
        gamma=0.99,
        noise_scale=0.1,
        polyak=0.995,
        target_update_step_interval=1,
        target_update_episode_interval=1
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

        self.noise_scale = noise_scale
        self.polyak = polyak

        self.act_dim = action_space.shape[0]
        self.act_range = 1 # 全ての行動空間において同じであることを想定、複数指定は今後対応予定
        self.target_update_step_interval = target_update_step_interval
        self.target_update_episode_interval = target_update_episode_interval


    def act_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()
        
        self.replay_buffer.append_and_update(
            self.state,
            self.action,
            next_state,
            reward
        )
        self.replay()

        if self.target_update_step_interval and self.step % self.target_update_step_interval == 0:
            self._update_target_network()
            
        self.state = next_state
        self.action = self.explorer.explore(lambda: self._choice_noisy_action(next_state), lambda: self._choice_greedy_action(next_state))

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


    def _update_target_network(self):
        # targetの更新
        # polyak averaging
        with torch.no_grad():
            for p, p_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)


    def act_greedily(self, observation):
        return self._choice_greedy_action(observation)


    def _choice_greedy_action(self, observation):
        action = self.ac.act(torch.as_tensor(observation, dtype=torch.float32))
        return np.clip(action, -self.act_range, self.act_range)


    def _choice_noisy_action(self, observation):
        action = self._choice_greedy_action(observation)
        action += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(action, -self.act_range, self.act_range)


    def stop_episode_and_train(self, observation, reward):
        """
        エピソード終了時の処理(学習も行う)

        Args:
            observation (ndarray): 観測した状態
            reward (float):        即時報酬
        """
        next_state = torch.tensor(observation).float()
        
        self.replay_buffer.append_and_update(
            self.state,
            self.action,
            next_state,
            reward
        )
        self.replay()
        self.explorer.end_episode()
        
        self.action = None
        self.state = None

        self.episode += 1
        
        if self.target_update_episode_interval and self.episode % self.target_update_episode_interval == 0:
            self._update_target_network()

