import numpy as np
from gym.spaces.box import Box

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
        
    #discretized = np.array([], dtype=int)
    #for i, state in enumerate(observation):
    #    discretized = np.append(discretized, digitize_state(i, state))

    discretized = np.array([digitize_state(i, state) for i, state in enumerate(observation)])
    

    return generate_state_index(discretized, discretize_nums)
