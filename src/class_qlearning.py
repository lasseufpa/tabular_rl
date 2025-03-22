import numpy as np
import gymnasium as gym
import src.finite_mdp_utils as fmdp
from typing import Tuple, List
from src.finite_mdp_utils import ValueFunctionFromQtable as qt2vf
from src.finite_mdp_utils import convert_action_values_into_policy as getPol



class Qlearning_agent():
    def __init__(self, env: gym.Env,
                       episodes_per_run=100,
                       max_num_time_steps_per_episode=10,
                       num_runs=1, discountGamma=0.9,
                       stepSizeAlpha=0.1, explorationProbEpsilon=0.01,
                       possible_actions_per_state=None, verbosity=1):
        
        
        Q_table, hist = q_learning_several_episodes(env, episodes_per_run,
                                                    max_num_time_steps_per_episode,
                                                    num_runs, discountGamma,
                                                    stepSizeAlpha, explorationProbEpsilon,
                                                    possible_actions_per_state, verbosity)

        self.Q_table = Q_table
        self.hist = hist 
        self.vf = None
        self.deterministic_policy = None 

    def get_vf(self):
        self.vf = qt2vf(self.Q_table) 
        return self.vf
    
    def get_policy(self):
        self.deterministic_policy = getPol(self.Q_table)
        return self.deterministic_policy



def q_learning_episode(env: gym.Env,
                       stateActionValues: np.ndarray,
                       possible_actions_per_state: list,
                       max_num_time_steps=100, stepSizeAlpha=0.1,
                       explorationProbEpsilon=0.01, discountGamma=0.9) -> int:
    '''    
    An episode with Q-Learning. We reset the environment.
    Note stateActionValues is not reset within this method, and can be already initialized.
    One needs to pay attention to allowing only
    valid actions. The simple algorithms provided in Sutton's book do not
    take that in account. See the discussion:
    https://ai.stackexchange.com/questions/31819/how-to-handle-invalid-actions-for-next-state-in-q-learning-loss
    Here we use possible_actions_per_state to indicate the valid actions.
    This list is a member variable of KnownDynamicsEnv enviroments but must be generated
    for other enviroments with the method @get_unrestricted_possible_actions_per_state
    or a similar one, which takes in account eventual restrictions.
    @return: average reward within this episode
    '''
    env.reset()
    currentState = env.get_obs()
    rewards = 0.0
    for numIterations in range(max_num_time_steps):
        
        currentAction = fmdp.action_via_epsilon_greedy(currentState, stateActionValues,
                                                  possible_actions_per_state,
                                                  explorationProbEpsilon=explorationProbEpsilon,
                                                  run_faster=False)
        
        newState, reward, gameOver, truncade, history = env.step(currentAction)
        rewards += reward
        nstate = env.get_obs() 
        # Q-Learning update
        stateActionValues[currentState, currentAction] += stepSizeAlpha * (
            reward + discountGamma * np.max(stateActionValues[nstate, :]) -
            stateActionValues[currentState, currentAction])
        
        currentState = nstate
        if gameOver:
            break
    # normalize rewards to facilitate comparison
    return rewards / (numIterations+1)

def q_learning_several_episodes(env: gym.Env,
                                episodes_per_run=100,
                                max_num_time_steps_per_episode=10,
                                num_runs=1, discountGamma=0.9,
                                stepSizeAlpha=0.1, explorationProbEpsilon=0.01,
                                possible_actions_per_state=None, verbosity=1) -> tuple[np.ndarray, np.ndarray]:
    '''Use independent runs instead of a single run.
    Increase num_runs if you want smooth numbers representing the average.
    @return tuple with stateActionValues corresponding to best average rewards among all runs
    and '''
    if verbosity > 0:
        print("Running", num_runs, " runs of q_learning_several_episodes() with",
              episodes_per_run, "episodes per run")


    possible_actions_per_state = fmdp.get_unrestricted_possible_actions_per_state(env)

    # shows convergence over episodes
    rewardsQLearning = np.zeros(episodes_per_run)
    best_stateActionValues = np.zeros((env.S, env.A))  # store best over runs
    largest_reward = -np.inf  # initialize with negative value
    for run in range(num_runs):
        stateActionValues = np.zeros((env.S, env.A))  # reset for each run
        sum_rewards_this_run = 0
        for i in range(episodes_per_run):
            # update stateActionValues in-place (that is, updates stateActionValues)
            reward = q_learning_episode(env, stateActionValues,
                                        possible_actions_per_state,
                                        max_num_time_steps=max_num_time_steps_per_episode,
                                        stepSizeAlpha=stepSizeAlpha, discountGamma=discountGamma,
                                        explorationProbEpsilon=explorationProbEpsilon)
            rewardsQLearning[i] += reward
            sum_rewards_this_run += reward
        average_reward_this_run = sum_rewards_this_run / episodes_per_run
        if average_reward_this_run > largest_reward:
            largest_reward = average_reward_this_run
            best_stateActionValues = stateActionValues.copy()
            if verbosity > 0:
                print("Found better stateActionValues")
        if verbosity > 0:
            print('run=', run, 'average reward=', average_reward_this_run)
    # need to normalize to get rewards convergence over episodes
    rewardsQLearning /= num_runs
    if verbosity > 1:
        print('rewardsQLearning = ', rewardsQLearning)
        print('newStateActionValues = ', stateActionValues)
        print('qlearning_policy = ',
              fmdp.convert_action_values_into_policy(stateActionValues))
    return best_stateActionValues, rewardsQLearning