from models4rl.agents.base_agent import BaseAgent
from models4rl.replay_buffers.data_memory.data_memory import DataMemory
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(BaseAgent):
    """
    Advanced ActorCritic
    エントロピー項なし
    """
    def __init__(self, action_space, ac_network, p_optimizer, q_optimizer, criterion=F.smooth_l1_loss, n_step=5, gamma=0.99):
        """
        ac_networkはnetwork_utilsから定義
        """
        super().__init__(action_space, None, gamma)
        self.ac_network = ac_network
        self.criterion = criterion

        self.p_optimizer = p_optimizer
        self.q_optimizer = q_optimizer

        self.trans_memory = DataMemory() # トランジションを保存するメモリ、ExperienceMemoryは経験再生用なのでこちらを使用

        self.state = None
        self.action = None
        self.episode = 1
        self.step = 1
        self.n_step = n_step


    def act_and_train(self, observation, reward):
        next_state = torch.tensor(observation).float()

        self._append_memory(
            self.state,
            self.action,
            next_state,
            reward
        )

        if self.step % self.n_step == 0:
            self.update()
        
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

        self.update(True)

        self.state = None
        self.action = None

        self.episode += 1


    def update(self, is_terminal=False):
        if len(self.trans_memory) == 0:
            return


        if is_terminal:
            Q = 0
        else:
            terminal = self.trans_memory.get_latest_memories(1)[0]     
            Q = self.ac_network.q(terminal["next_state"])

        actor_loss = 0
        critic_loss = 0
        R = 0
        # 計算式:https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients
        # Rを逆から計算しているのがポイント
        for trans in self.trans_memory.get_latest_memories():
            r = trans["reward"]
            s = trans["state"]
            v = self.ac_network.q(s)
            action_prob = self.ac_network.p(s)[trans["action"]]

            Q = r + self.gamma * Q
            advantage = Q - v
            critic_loss += self.criterion(v, torch.tensor(Q))
            actor_loss -= torch.log(action_prob) * advantage

        critic_loss = critic_loss/len(self.trans_memory)
        actor_loss = actor_loss/len(self.trans_memory)
        
        self.p_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.p_optimizer.step()
        
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()
        
        self.trans_memory.reset() # on-policyなのでpolicyが更新されたらデータを破棄


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