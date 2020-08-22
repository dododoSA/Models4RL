from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np
from typing import List, Union

def bins(clip_min:float, clip_max:float, num:int) -> np.ndarray:
    return np.linspace(clip_min, clip_max, num+1)[1:-1]

def generate_state_index(discretized, discretized_nums) -> int:
    index = 0
    for i, x in enumerate(discretized):
        digit = 1
        for j in range(i):
            digit *= discretized_nums[j]
        index += x * digit

    return index


def discretize_Box_state(observation:np.ndarray, observation_space:Box, discretize_nums:np.ndarray) -> int:
    def digitize_state(i, state):
        return np.digitize(state, bins=bins(observation_space.low[i], observation_space.high[i], discretize_nums[i]))
        
    discretized = np.array([], dtype=int)
    for i, state in enumerate(observation):
        discretized = np.append(discretized, digitize_state(i, state))
    # discretized = map(digitize_state, ) 出来ればmapで実装したい

    return generate_state_index(discretized, discretize_nums)


class Qlearning():
    def __init__(
                    self, 
                    discretize_nums:Union[int, List[int], np.ndarray], 
                    observation_space:Union[Box], 
                    action_space:Discrete, 
                    alpha:float=0.5, gamma:float=0.99, 
                    init_q_max:float=1.0
                ):
        """
        コンストラクタ

        Args:
            discritize_num (int or list of int): 状態の次元ごとに分割数を指定。整数値が与えられた場合、全て値の等しいlistが与えられることと同じ。
            observation_space (Box):             環境の状態空間 env.observation_spaceをそのまま渡せばOK、Boxのみ可
            action_space (Discrete):             行動空間 env.action_spaceをそのまま渡せばOK、一次元離散値
            alpha (float):                       学習率,初期値0.5
            gamma (float):                       割引率, 初期値0.99
            init_q_max (float)                   q_tableの初期値のしきい値 |q| <= init_q_max
        """
        self.discretize_nums = discretize_nums
        self.observation_space = observation_space
        state_num = observation_space.shape[0] # self付けるかどうか問題
        self.action_space = action_space
        self.action_num = action_space.n
        self.alpha = alpha
        self.gamma = gamma

        self.state = None
        self.action = None
        self.episode = 0

        if type(discretize_nums) == int:
            self.discretize_nums = np.full((state_num), discretize_nums)
        elif type(discretize_nums) == list:
            self.discretize_nums = np.array(discretize_nums)


        assert len(self.discretize_nums) == state_num, 'discretize_nums and observation_space must have the same length.'

        self.all_state_num = 1
        for i in range(state_num):
            self.all_state_num *= self.discretize_nums[i]

        self.q_table = np.random.uniform(low=-init_q_max, high=init_q_max, size=(self.all_state_num, self.action_num))

    def act_and_train(self, observation:np.ndarray, reward:float) -> int:
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
        next_state = discretize_Box_state(observation, self.observation_space, self.discretize_nums)

        if self.state != None and self.action != None:
            self.update_q_table(reward, next_state)
            
        # 行動を起こす前の状態と、起こした行動を保存
        self.state = next_state
        self.action = self.act_by_epsilon_greedy(next_state)

        return self.action

    def update_q_table(self, reward, next_state) -> None:
        """
        Q関数の更新

        Args:
            reward (float): 前回行った行動による報酬
            next_state: インデックス化された前回の行動による移り先の状態(現在の状態)

        Returns:
            None
        """
        td_error = reward + self.gamma*np.max(self.q_table[next_state]) - self.q_table[self.state, self.action]
        self.q_table[self.state, self.action] += self.alpha*td_error


    def act_by_epsilon_greedy(self, next_state) -> int:
        """
        ε-greedy法に基づく探索
        ここはクラス化してコンストラクタから指定する可能性あり
        """
        epsilon = 1 / (self.episode + 1)
        self.episode += 1

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[next_state])
        else:
            action = self.action_space.sample()
        
        return action

    def act_greedy(self, observation):
        next_state = discretize_Box_state(observation, self.observation_space, self.discretize_nums)
        return np.argmax(self.q_table[next_state])

    def stop_episode_and_train(self, observation, reward):
        next_state = discretize_Box_state(observation, self.observation_space, self.discretize_nums)
        self.update_q_table(reward, next_state)
