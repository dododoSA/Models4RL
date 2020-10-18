from models4rl.agents.base_agent import BaseAgent
from models4rl.replay_buffers.replay_buffer import ReplayBuffer
from models4rl.explorers.base_explorer import BaseExplorer

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import copy
from gym.spaces.discrete import Discrete



class DDQN(BaseAgent):
    def __init__(
        self,
        action_space:Discrete,
        q_network:Module,
        optimizer:Optimizer,
        criterion:Module,
        explorer:BaseExplorer,
        replay_buffer:ReplayBuffer,
        batch_size:int=32,
        gamma:float=0.99,
        target_update_step_interval:int=0,
        target_update_episode_interval:int=5
    ):
        """
        コンストラクタ

        Args:
            action_space (Discrete):              行動空間 env.action_spaceをそのまま渡せばOK、一次元離散値
            q_network (Module):                   学習するネットワーク 型はとりあえず親クラスを書いている
            optimizer (Optimizer):                最適化アルゴリズム
            criterion (Module):                   損失関数 型はとりあえず自作するときの親クラスを書いている
            explorer (BaseExplorer):              探索アルゴリズム
            replay_buffer (ReplayBuffer):         経験再生用のメモリ
            batch_size (int):                     経験再生からリプレイを行う際にサンプリングする数, 初期値32 (replay bufferに含めてもいいかもしれない)
            gamma (float):                        割引率, 初期値0.99
            target_update_step_interval (int):    学習の目標となるQNetworkを何ステップ毎に更新するか. 0の場合は更新されない. 1ステップが終了するとカウントはリセットされる. 初期値0
            target_update_episode_interval (int): 学習の目標となるQNetworkを何エピソード毎に更新するか. 初期値5
        """

        assert target_update_step_interval >= 0, 'target_update_step_interval must be positive or 0.'
        assert target_update_episode_interval > 0, 'target_update_episode_interval must be positive.'

        super().__init__(action_space, explorer, gamma)

        self.q_network = q_network
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.target_network = copy.deepcopy(self.q_network)

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size


        self.state = None
        self.action = None
        self.target_update_step_interval = target_update_step_interval
        self.target_update_episode_interval = target_update_episode_interval
        self.episode = 1
        self.step = 1


    def act_and_train(self, observation, reward):
        """
        学習と行動を行う関数。引数として渡された状態と報酬は一度保存し、次のステップで使用される。

        Args:
            observation (numpy.array): 前回行った行動による移り先の状態(現在の状態)
            reward (float): 前回行った行動による報酬
        
        Returns:
            int: 移り先の状態にて起こす行動

        Examples:
            # init
            obs = env.reset()
            reward = 0
            done = False

            ...

            # run 必ずこの順番で呼ぶ
            action = agent.act_and_train(obs, reward)
            obs, reward, done, info = env.step(action)

        """
        next_state = torch.tensor(observation).float()
        
        self.replay_buffer.append(self.state, self.action, next_state, reward)
        self.replay()

        if self.target_update_step_interval and self.step % self.target_update_step_interval == 0:
            self._update_target_network()

        self.state = next_state
        self.action = self.explorer.explore(self.action_space.sample, lambda: self._choice_greedy_action(next_state))

        self.step += 1
        return self.action


    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, next_state_batch, action_batch, reward_batch = self.replay_buffer.sample(self.batch_size).values()
        

        self.q_network.eval()

        q_values = self.q_network(state_batch).gather(1, action_batch)
        tmp = self.q_network(next_state_batch)
        _, next_max_actions = tmp.max(1, keepdim=True) # ここに追加
        next_q_values = torch.zeros_like(q_values, dtype=float)


        self.target_network.eval()
        
        next_q_values = self.target_network(next_state_batch)# .gather(1, next_max_actions) # ここを修正
        next_q_values = next_q_values.gather(1, next_max_actions).squeeze(1)
        target_values = self.gamma * next_q_values + reward_batch


        self.q_network.train()
        
        loss = self.criterion(q_values, target_values.unsqueeze(1).float())
        self.optimizer.zero_grad()
        loss.backward()

        #for p in self.q_network.parameters():
        #    p.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def _update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


    def _choice_greedy_action(self, observation):
        self.q_network.eval()
        with torch.no_grad():
            action = self.q_network(observation).max(0)[1].item()
        return action


    def act_greedily(self, observation):
        next_state = torch.tensor(observation).float()
        return self._choice_greedy_action(next_state)


    def stop_episode_and_train(self, observation, reward):
        """
        エピソード終了時の処理(学習も行う)

        Args:
            observation (ndarray): 観測した状態
            reward (float):        即時報酬
        """
        next_state = torch.tensor(observation).float()
        self.replay_buffer.append(self.state, self.action, next_state, reward)
        self.replay()
        self.explorer.end_episode()
        
        self.action = None
        self.state = None

        self.episode += 1
        
        if self.episode % self.target_update_episode_interval == 0:
            self._update_target_network()
