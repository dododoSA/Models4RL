from models4rl.explorers.base_explorer import BaseExplorer

from abc import ABCMeta, abstractmethod
import numpy as np
from gym.spaces.discrete import Discrete

class BaseAgent(metaclass=ABCMeta):
    def __init__(self, action_space:Discrete, explorer:BaseExplorer, gamma:float=0.99):
        self.action_space = action_space
        self.explorer = explorer
        self.gamma = gamma

        self.episode = 1
        self.step = 1


    @abstractmethod
    def act_and_train(self) -> int:
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
        raise NotImplementedError()


    @abstractmethod
    def _choice_greedy_action(self, observation:np.ndarray) -> int:
        raise NotImplementedError()


    @abstractmethod
    def act_greedily(self, observation:np.ndarray) -> int:
        # _choice_greedy_actionを呼び出すだけだから抽象関数にしなくてもいいかもしれない
        raise NotImplementedError()


    @abstractmethod
    def stop_episode_and_train(self, observation:np.ndarray, reward:float) -> None:
        """
        エピソード終了時の処理(学習も行う)

        Args:
            observation (ndarray): 観測した状態
            reward (float):        即時報酬
        """
        raise NotImplementedError()