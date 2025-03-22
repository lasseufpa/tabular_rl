import numpy as np
from src.knowm_dynamics_env import VerboseKnownDynamicsEnv as vkde
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

print(rewardsTable.shape)

# Number of states and actions 
Ns = nextStateProbability.shape[0]
Na = nextStateProbability.shape[1]

# Treshold that manage which structure class use
sparsity_treshold = 0.1

# Creatiing state information
dictionaryGetIndex = {
    (0, 0, 0) : 1,
    (1, 2, 3) : 2,
    (5, 9, 7) : 3

}

listGivenIndex = [(0, 0, 0), (1, 2, 3), (5, 9, 7) ]

name_components = ["sensor 0", "sensor 1", "sensor 2"]

state_inf = [dictionaryGetIndex, listGivenIndex, name_components]


# Creatiing action information
dictionaryGetIndex = {
    ("up"): 1,
    ("down") : 2,

}

listGivenIndex = [("up"), ("down")]

name_components = ["act"]

act_inf = [dictionaryGetIndex, listGivenIndex, name_components]

# Creating the verbose env
verb_env = vkde(nextStateProbability, rewardsTable, Ns, Na, sparsity_treshold, state_inf, act_inf)

# Creating uniform policy
policy = fmdp.get_uniform_policy_for_fully_connected(verb_env.S, verb_env.A)

# Verbose print about the policy
verb_env.pretty_print_policy(policy)

print(verb_env.observation_space)