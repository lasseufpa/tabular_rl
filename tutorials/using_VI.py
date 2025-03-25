import numpy
from envs.sutton_grid_world_env import SuttonGridWorldEnv
import src.finite_mdp_utils as fmdp
from src.class_vi import VI_agent

env = SuttonGridWorldEnv()
gamma = 0.9
tolerance = 1e-4

Vi_agent = VI_agent(env, gamma, tolerance)

print(Vi_agent.Q_table)
print(Vi_agent.get_vf())
print(Vi_agent.get_policy())

env.pretty_print_policy(Vi_agent.deterministic_policy)
