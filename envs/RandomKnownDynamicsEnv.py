from known_dynamics_env import KnownDynamicsEnv
import numpy as np

class RandomKnownDynamicsEnv(KnownDynamicsEnv):
    '''
    Initialize a matrix with probability distributions.
    '''

    def __init__(self, S: int, A: int):
        nextStateProbability = self.init_random_next_state_probability(S, A)
        # these rewards can be negative
        rewardsTable = np.random.randn(S, A, S)
        self.__version__ = "0.1.0"
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def init_random_next_state_probability(self, S: int, A: int) -> np.ndarray:
        nextStateProbability = np.random.rand(S, A, S)  # positive numbers
        # for each pair of s and a, force numbers to be a probability distribution
        for s in range(S):
            for a in range(A):
                sum = np.sum(nextStateProbability[s, a])
                if sum == 0:
                    raise Exception(
                        "Sum is zero. np.random.rand did not work properly?")
                nextStateProbability[s, a] /= sum
        return nextStateProbability
    
if __name__ == "__main__":
    env = RandomKnownDynamicsEnv(4,5)