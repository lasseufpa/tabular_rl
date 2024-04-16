'''
Implemetation of SARSA algorithm

Suports SuttonGridWorldEnv and SimpleKnownDynamicsEnv

'''

from tabular_rl.envs.sutton_grid_world_env import SuttonGridWorldEnv
from tabular_rl.envs.simple_known_dynamics_env import SimpleKnownDynamicsEnv

import gym
import numpy as np
import random

#Define SARSA global parameters
alpha = 0.1  #Learning rate
gamma = 0.9  #Discount factor
epsilon = 0.1  #Exploration rate

def set_alpha(a):
  global alpha
  alpha = a
  
def set_gamma(g):
  global gamma
  gamma = g
  
def set_epsilon(e):
  global epsilon
  epsilon = e

def create_q_table(env):
  """
  Creates a Q-table with zeros for all states and actions.
  """
  state_space = env.observation_space.n
  action_space = env.action_space.n
  return np.zeros((state_space, action_space))

def choose_action(q_table, state, epsilon):
  """
  Selects the next action based on epsilon-greedy strategy.
  """
  if random.random() < epsilon:
    #Explore: choose a random action
    return env.action_space.sample()
  else:
    #Exploit: choose the action with the highest Q-value
    return np.argmax(q_table[state])

def sarsa(env, num_episodes, max_steps_per_episode):
  """
  Implements the SARSA algorithm for the given environment.
  """
  q_table = create_q_table(env)

  for episode in range(num_episodes):
    #Reset the environment
    state = env.reset()

    for step in range(max_steps_per_episode):
      #Choose action with epsilon-greedy strategy
      action = choose_action(q_table, state, epsilon)

      #Take action and observe reward and next state
      next_state, reward, done, _ = env.step(action)

      #Choose the next action (for SARSA)
      next_action = choose_action(q_table, next_state, epsilon)

      #Update Q-table with SARSA rule
      q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

      #Update state for the next iteration
      state = next_state

      if done:
        break

  return q_table



if __name__ == "__main__": 
  
  #Parse command-line arguments
  import argparse
  parser = argparse.ArgumentParser(description="SARSA for custom Environment")
  parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
  parser.add_argument("--max_steps_per_episode", type=int, default=200, help="Maximum steps per episode")
  parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (alpha)")
  parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (gamma)")
  parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate (epsilon)")
  parser.add_argument("--env", type=int, default=0, help="Environment (0 = SuttonGridWorld, 1 = SimpleKnownDynamics) (Default = 0)")
  args = parser.parse_args()
   
  #Define the custom environment:
  if (args.env == 1):
    env = SimpleKnownDynamicsEnv() #custom environment with known dynamics
  else:
    env = SuttonGridWorldEnv() #gridworld    
  
  #Define SARSA parameters
  set_alpha(args.alpha)
  set_gamma(args.gamma)
  set_epsilon(args.epsilon)

  #Define training parameters
  num_episodes = args.num_episodes
  max_steps_per_episode = args.max_steps_per_episode

  #Train the agent using SARSA
  q_table = sarsa(env, num_episodes, max_steps_per_episode)
  
  print('\nSARSA parametrs:\n\nLearning rate: ', alpha)
  print('Discount factor: ', gamma)
  print('Exploration rate: ', epsilon)
  print('\nNumber of episodes: ', num_episodes)
  print('Max steps per episode: ', max_steps_per_episode)
  print('\nEnvironment: ', type(env))
  print('Q-table:\n')
  print(q_table)