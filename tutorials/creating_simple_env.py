import numpy as np
from src.knowm_dynamics_env import KnownDynamicsEnv as kde
import src.finite_mdp_utils as fmdp

# probabilitys p(s'/s,a)
nextStateProbability = np.array([[[0.5, 0.5, 0],
                                    [0.9, 0.1, 0]],
                                    [[0, 0.5, 0.5],
                                    [0, 0.2, 0.8]],
                                    [[0, 0, 1],
                                    [0, 0, 1]]])

# rewards r(s', a, s)
rewardsTable = np.array([[[-3, 0, 0],
                            [-2, 5, 5]],
                            [[4, 5, 0],
                            [2, 2, 6]],
                            [[-8, 2, 80],
                            [11, 0, 3]]])

# Number of states and actions 
Ns = nextStateProbability.shape[0]
Na = nextStateProbability.shape[1]

# Treshold that manage which structure class use
sparsity_treshold = 0.1

# Creating the environment
env = kde(nextStateProbability, rewardsTable, Ns, Na, sparsity_treshold)

# Generating a uniform policy
policy = fmdp.get_uniform_policy_for_fully_connected(env.S, env.A)

# Generating a trajectory with 100 steps
trajectory = fmdp.generate_trajectory(env, policy, num_steps=100)