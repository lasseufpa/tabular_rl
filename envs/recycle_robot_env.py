from known_dynamics_env import KnownDynamicsEnv
import numpy as np

class RecycleRobotEnv(KnownDynamicsEnv):
    def __init__(self):
        self.__version__ = "0.1.0"
        nextStateProbability, rewardsTable = self.recycle_robot_matrices()
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def recycle_robot_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        S = 2  # number of states
        A = 3  # number of actions
        alpha = 0.1
        beta = 0.4
        rsearch = -1
        rwait = -2
        # states
        high = 0
        low = 1
        # actions
        search = 0
        wait = 1
        recharge = 2

        # nextStateProbability Table
        nextStateProbability = np.zeros([S, A, S])
        nextStateProbability[high, search, high] = alpha
        nextStateProbability[high, search, low] = 1-alpha
        nextStateProbability[low, search, high] = 1 - beta
        nextStateProbability[low, search, low] = beta
        nextStateProbability[high, wait, high] = 1
        nextStateProbability[high, wait, low] = 0
        nextStateProbability[low, wait, high] = 0
        nextStateProbability[low, wait, low] = 1
        nextStateProbability[low, recharge, high] = 1
        nextStateProbability[low, recharge, low] = 0

        rewardsTable = np.zeros([S, A, S])  # these rewards can be negative
        rewardsTable[high, search, high] = rsearch
        rewardsTable[high, search, low] = rsearch
        rewardsTable[low, search, high] = -3
        rewardsTable[low, search, low] = rsearch
        rewardsTable[high, wait, high] = rwait
        rewardsTable[low, wait, low] = rwait

        return nextStateProbability, rewardsTable
    
if __name__ == "__main__":
    env = RecycleRobotEnv()