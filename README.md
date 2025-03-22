# tabular_rl
Tabular RL is a Python library designed for finite Markov decision processes (MDPs) and tabular reinforcement learning with known dynamics. It provides efficient algorithms and utilities for solving MDPs, estimating optimal policies, and experimenting with various RL environments.

# Installation
To install the repository, first clone it and in a folder, then use:
``pip install -e tabular_rl/``

with that, in a .py, you can call the module with
```
import tabular_rl 
```
# Quick Start

Simple use of the module

``` 
from tabular_rl import finite_mdp_utils as fmdp
from tabular_rl import vi_agent as VI
from tabular_rl import  simple_known_dynamics

# Generate a simple environment
env = simple_known_dynamics.SimpleKnownDynamicsEnv()
gamma =  0.9
tolerance = 0.0001
debug =  False

# Run the VI algorithm and get the policy
Vi_agent = VI.VI_agent(env, gamma, tolerance, debug)
vi_policy = Vi_agent.get_policy()

# Generate some episodes with VI policy in that env
sum_rewards_episode = fmdp.run_several_episodes(env, Vi_agent.get_policy())
```

# Composition
This module is splited in two main parts, environments (envs) and codes for agent and experiments (source, src).

The src is composed by:
- **known dynamics env**  
  It contains the main classes to represent MDPs with known dynamics as a gym environment.
  - **class Known dynamic env**
    
    It contains the information of current state, number of steps and manage the type of struct used to store the env dynamics and compose the step and reset methods.
    
  - **Verbose Known dynamic env**

    It put a layer of informations above the class Known dynamic env, used to integrate with deep RL library as Stable Baseline 3. It requires, besides the env dynamics, structures contained informations about state and actions
    
  - **simple struct (for now is listKdEnv)**

    class to run the enviroment on simplified structures, like lists. It works better with environments with low dynamics sparsity
    
  - **Optimized struct (for now is dictKdEnv)**

     class to run the enviroment on optimized structures based on dictionarys.

- **VI agent**  
  Class to manage how it will run the Value Iteration algorithm based on the type of strucutre chosed by known dynamics env

- **Qlearning agent**  
  Run a simple Q learning agent

- **FMDP Utils**  
  Some RL utilites like make a trajectory based on a environment and a policy, calculate theoretical values of Action Value Fuctions (just work for list dynamics), convert AVF into policy...  

- **Mobile Utils**  
  Some utilites to create the envs multiband_scheduling and user_schedulling

- **Optimum Values**  
  I hope my boss allows me to remove it.
