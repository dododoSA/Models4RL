import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box

def bins(clip_min:float, clip_max:float, num:int):
    return np.linspace(clip_min, clip_max, num+1)[1:-1]

def generate_state_index(discretized, discretized_nums):
    index = 0
    for i, x in enumerate(discretized):
        digit = 1
        for j in range(i):
            digit *= discretized_nums[j]
        index += x * digit

    return index


def discretize_Box(observation:np.ndarray, observation_space:Box, discretize_nums:np.ndarray):
    def digitize_state(i, state):
        return np.digitize(state, bins=bins(observation_space.low[i], observation_space.high[i], discretize_nums[i]))
        
    discretized = map(digitize_state, enumerate(observation))

    return generate_state_index(discretized, discretize_nums)


class Qlearnng():
    def __init__(
                    self, 
                    discretize_nums:Union[int, List[int], np.ndarray], 
                    observation_space:Union[Box, Discrete, np.ndarray], 
                    action_space:Discrete, 
                    alpha:float=0.5, gamma:float=0.99, 
                    init_q_max:float=1.0
                ):
        """
        コンストラクタ

        Args:
            discritize_num (int or list of int): 状態の次元ごとに分割数を指定。整数値が与えられた場合、全て値の等しいlistが与えられることと同じ。
            observation_space (Space):           環境の状態空間 env.observation_spaceをそのまま渡せばOK
            action_space (Discrete):             行動空間 env.action_spaceをそのまま渡せばOK
            alpha (float):                       学習率,初期値0.5
            gamma (float):                       割引率, 初期値0.99
            init_q_max (float)                   q_tableの初期値のしきい値 |q| <= init_q_max
        """
        self.discretise_nums = discretize_nums
        self.observation_space = observation_space
        self.state_num = observation_space.shape[0]
        self.action_space = action_space
        self.action_num = action_space.n
        self.alpha = alpha
        self.gamma = gamma

        if type(discretize_nums) == int:
            self.discretize_nums = np.full((state_num), discretize_nums)
        elif type(discretize_nums) == list:
            self.discretise_nums = np.ndarray(discretise_nums)


        assert len(discretize_nums) == self.state_num, 'discretize_nums and observation_space must have the same length.'

        self.all_state_num = 1
        for i in range(state_num):
            self.all_state_num *= discretize_nums[i]

        self.q_table = np.random.uniform(low=-init_q_max, high=init_q_max, size=(self.all_state_num, self.action_num))

    def update_q_table(self):
        pass