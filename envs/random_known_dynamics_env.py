from src.knowm_dynamics_env import KnownDynamicsEnv
import numpy as np
import random
from src import finite_mdp_utils as fmdp


class RandomKnownDynamicsEnv(KnownDynamicsEnv):
    '''
    Initialize a matrix with probability distributions.
    '''

    def __init__(self, S: int, A: int, sparcity_rate:float, assert_struct:str):

        nextStateProbability = self.init_random_next_state_probability(S, A, sparcity_rate)
        # these rewards can be negative
        rewardsTable = 8*np.random.randn(S, A, S)
        self.__version__ = "0.1.0"
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable, assert_struct = 'default')

    def init_random_next_state_probability(self, S: int, A: int, sparcity_rate:float) -> np.ndarray:
        nextStateProbability = np.zeros([S,A,S])
        nelements = int(sparcity_rate*S)
        for s in range(S):
            for a in range(A):

                auxVec = random.sample(range(0, S), nelements)

                for i in range(len(auxVec)):
                    nextStateProbability[s,a,auxVec[i]] = abs(10*np.random.randn())
                
                nextStateProbability[s,a,:] /= sum(nextStateProbability[s,a,:])
            
        return nextStateProbability


if __name__ == "__main__":
    env = RandomKnownDynamicsEnv(4, 5)
    num_steps = 10
    uniform_policy = fmdp.get_uniform_policy_for_fully_connected(env.S, env.A)
    taken_actions, rewards_tp1, states = fmdp.generate_trajectory(
        env, uniform_policy, num_steps)
    trajectory = fmdp.format_trajectory_as_single_array(
        taken_actions, rewards_tp1, states)
    print("Complete trajectory vector:")
    print(trajectory)
    print("Interpret trajectory with print_trajectory() method:")
    fmdp.print_trajectory(trajectory)
