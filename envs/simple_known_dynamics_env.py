from known_dynamics_env import KnownDynamicsEnv
import numpy as np

class SimpleKnownDynamicsEnv(KnownDynamicsEnv):
    def __init__(self):
        self.__version__ = "0.1.0"
        # make it a "left-right" Markov process without skips:
        # the state index cannot decrease over time nor skip
        # over the next state
        nextStateProbability = np.array([[[0.5, 0.5, 0],
                                          [0.9, 0.1, 0]],
                                         [[0, 0.5, 0.5],
                                          [0, 0.2, 0.8]],
                                         [[0, 0, 1],
                                          [0, 0, 1]]])
        rewardsTable = np.array([[[-3, 0, 0],
                                  [-2, 5, 5]],
                                 [[4, 5, 0],
                                  [2, 2, 6]],
                                 [[-8, 2, 80],
                                  [11, 0, 3]]])
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def reset(self) -> int:
        # make sure initial state is 0
        super().reset()
        self.current_observation_or_state = 0
        return self.current_observation_or_state
    
if __name__ == "__main__":
    env = SimpleKnownDynamicsEnv()