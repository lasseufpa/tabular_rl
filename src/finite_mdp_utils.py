'''
This class implements methods to deal with 
a finite Markov Decision Process (MDP) for tabular RL.
Look at the definition of a finite MDP in page 49 of
Sutton & Barto's book, version 2020:

    Note that a policy is represented here as a distribution over the
    possible actions for each state and stored as an array of dimension S x A.
    A general policy may be represented in other ways. But considering the
    adopted representation here, the "policy" coincides with the distribution
    of actions for each state.

The methods are "static", in the sense that are not defined within a class.

Note that a policy is represented here as a matrix S x A, providing a distribution
over the possible actions for each state. A matrix with the state values can be
easily converted into a policy. 

Aldebaro. Oct 25, 2023.
'''
from __future__ import print_function
import numpy as np
import gymnasium as gym
from random import choices
from gymnasium import spaces
from src.knowm_dynamics_env import KnownDynamicsEnv, listKdEnv
from scipy.optimize import fsolve

def check_if_fmdp(environment: gym.Env):
    # checks if env is a FMDP gym with discrete spaces
    assert isinstance(environment.action_space, spaces.Discrete)
    assert isinstance(environment.observation_space, spaces.Discrete)


def get_space_dimensions_for_openai_gym(environment: gym.Env) -> tuple[int, int]:
    '''
    Returns S and A.
    '''
    assert isinstance(environment.action_space, spaces.Discrete)
    assert isinstance(environment.observation_space, spaces.Discrete)
    S = environment.observation_space.n
    A = environment.action_space.n
    return S, A


def compute_state_values(env: gym.Env, policy: np.ndarray, in_place=False,
                         discountGamma=0.9) -> tuple[np.ndarray, int]:
    '''
    Iterative policy evaluation. Page 75 of [Sutton, 2018].
    Here a policy (not necessarily optimum) is provided.
    It can generate, for instance, Fig. 3.2 in [Sutton, 2018]
    '''
    S = env.S
    A = env.A
    # (S, A, nS) = self.environment.nextStateProbability.shape
    # A = len(actionListGivenIndex)
    new_state_values = np.zeros((S,))
    state_values = new_state_values.copy()
    iteration = 1
    while True:
        src = new_state_values if in_place else state_values
        for s in range(S):
            value = 0
            for a in range(A):
                if policy[s, a] == 0:
                    continue  # save computation
                for nexts in range(S):
                    p = env.nextStateProbability[s, a, nexts]
                    r = env.rewardsTable[s, a, nexts]
                    value += policy[s, a] * p * \
                        (r + discountGamma * src[nexts])
                    # value += p*(r+discount*src[nexts])
            new_state_values[s] = value
        # AK-TODO, Sutton, end of pag. 75 uses the max of individual entries, while
        # here we are using the summation:
        improvement = np.sum(np.abs(new_state_values - state_values))
        # print('improvement =', improvement)
        if False:  # debug
            print('state values=', state_values)
            print('new state values=', new_state_values)
            print('it=', iteration, 'improvement = ', improvement)
        if improvement < 1e-4:
            state_values = new_state_values.copy()
            break

        state_values = new_state_values.copy()
        iteration += 1

    return state_values, iteration


def convert_action_values_into_policy(action_values: np.ndarray) -> np.ndarray:
    (S, A) = action_values.shape
    policy = np.zeros((S, A))
    for s in range(S):
        maxPerState = max(action_values[s])
        maxIndices = np.where(action_values[s] == maxPerState)
        # maxIndices is a tuple and we want to get first element maxIndices[0]
        # impose uniform distribution
        policy[s, maxIndices] = 1.0 / len(maxIndices[0])
    return policy


def get_uniform_policy_for_fully_connected(S: int, A: int) -> np.ndarray:
    '''
    Assumes all actions can be performed at each state.
    See @get_uniform_policy_for_known_dynamics for
    an alternative that takes in account the dynamics
    of the defined environment.
    '''
    policy = np.zeros((S, A))
    uniformProbability = 1.0 / A
    for s in range(S):
        for a in range(A):
            policy[s, a] = uniformProbability
    return policy


def run_several_episodes(env: gym.Env, policy: np.ndarray, num_episodes=10, max_num_time_steps_per_episode=100, printInfo=False, printPostProcessingInfo=False, seed = None) -> np.ndarray:
    '''
    Runs num_episodes episodes and returns their rewards.'''
    rewards = np.zeros(num_episodes)
    for e in range(num_episodes):
        rewards[e] = run_episode(env, policy, maxNumIterations=max_num_time_steps_per_episode,
                                 printInfo=printInfo, printPostProcessingInfo=printPostProcessingInfo, seed = seed)
    return rewards


def run_episode(env: gym.Env, policy: np.ndarray, maxNumIterations=100, printInfo=False, printPostProcessingInfo=False, seed = None) -> int:
    '''
    Reset and runs a complete episode for the environment env according to
    the specified policy.
    The policy already has distribution probabilities that specify the
    valid actions, so we do not worry about it (we do not need to invoke
    methods such as @choose_epsilon_greedy_action)
    '''
    env.reset()
    s = env.get_obs()
    totalReward = 0
    list_of_actions = np.arange(env.A)
    if printInfo:
        print('Initial state = ', env.stateListGivenIndex[s])
    for it in range(maxNumIterations):
        policy_weights = np.squeeze(policy[s])
        sumWeights = np.sum(policy_weights)
        if sumWeights == 0:
            print(
                "Warning: reached state that does not have a valid action to take. Ending the episode")
            break
        action = choices(list_of_actions, weights=policy_weights, k=1)[0]
        ob, reward, gameOver,truncade, history = env.step(action)

        # assume the user may want to apply some postprocessing step, similar to a callback function
        #env.postprocessing_MDP_step(history, printPostProcessingInfo)
        if printInfo:
            print(history)
        totalReward += reward
        s = env.get_obs()  # update current state
        if gameOver == True:
            break
    if printInfo:
        print('totalReward = ', totalReward)
    return totalReward


def get_unrestricted_possible_actions_per_state(env: gym.Env) -> list:
    '''
    Create a list of lists, that indicates for each state, the list of allowed actions.
    Here we indicate all actions are valid for each state.
    '''
    S = env.S
    A = env.A
    possible_actions_per_state = list()
    for s in range(S):
        possible_actions_per_state.append(list())
        for a in range(A):
            possible_actions_per_state[s].append(a)
    return possible_actions_per_state


def action_via_epsilon_greedy(state: int,
                              stateActionValues: np.ndarray,
                              possible_actions_per_state: list,
                              explorationProbEpsilon=0.01, run_faster=False) -> int:
    '''
    Choose an action based on epsilon greedy algorithm.
    '''
    
    if np.random.binomial(1, explorationProbEpsilon) == 1:
        # explore among valid options
        return np.random.choice(possible_actions_per_state[int(state)])
    else:
        # exploit, choosing an action with maximum value
        return action_greedy(state, stateActionValues, possible_actions_per_state, run_faster=run_faster)


def action_greedy(state: int,
                  stateActionValues: np.ndarray,
                  possible_actions_per_state: list,
                  run_faster=False) -> int:
    '''
    Greedly choose an action with maximum value.
    '''
    values_for_given_state = stateActionValues[int(state)]
    if run_faster == True:
        # always return the first index with maximum value
        # this may create a problem for the agent to explore other options
        # or choose an invalid option
        max_index = np.argmax(values_for_given_state)
    else:
        actions_for_given_state = possible_actions_per_state[int(state)]
        # make sure the action is valid, but keeping invalid actions with value=-infinity
        valid_values_for_given_state = -np.inf * \
            np.ones(values_for_given_state.shape)
        valid_values_for_given_state[actions_for_given_state] = values_for_given_state[actions_for_given_state]
        max_value = np.max(valid_values_for_given_state)
        # Use numpy.where to get all indices where the array is equal to its maximum value
        all_max_indices = np.where(
            valid_values_for_given_state == max_value)[0]
        max_index = np.random.choice(all_max_indices)
    return max_index





def create_random_next_state_probability(S: int, A: int) -> np.ndarray:
    nextStateProbability = np.random.rand(S, A, S)  # positive numbers
    for s in range(S):
        for a in range(A):
            pmf = nextStateProbability[s, a]  # probability mass function (pmf)
            total_prob = sum(pmf)
            if total_prob == 0:
                # arbitrarily, make first state have all probability
                nextStateProbability[s, a, 0] = 1
            else:
                # normalize to have a pmf
                nextStateProbability[s, a] /= total_prob
    return nextStateProbability


def generate_trajectory(env: gym.Env, policy: np.ndarray, num_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    We generate trajectories according to the specified policy.
    Note that actions are generated by the agent, not by the environment. 
    At time t=0 one obtains the reward for t=1 (convention of Sutton's book).
    We do not reset the environment env. It starts from its current state.
    The policy already has distribution probabilities that specify the
    valid actions, so we do not worry about it (we do not need to invoke
    methods such as @choose_epsilon_greedy_action)
    '''
    env.reset()
    list_of_actions = np.arange(env.A)  # list all existing actions
    # initialize arrays
    taken_actions = np.zeros(num_steps, dtype=int)
    rewards_tp1 = np.zeros(num_steps)
    states = np.zeros(num_steps, dtype=int)
    for t in range(num_steps):
        states[t] = env.get_obs()  # current state
        # choose action according to the policy
        taken_actions[t] = choices(
            list_of_actions, weights=policy[states[t]])[0]
        ob, reward, gameOver,truncated, history = env.step(taken_actions[t])
        print(history)
        # at time t=0 one obtains the reward for t=1 (convention of Sutton's book)
        rewards_tp1[t] = reward
    return taken_actions, rewards_tp1, states


def format_trajectory_as_single_array(taken_actions: np.ndarray, rewards_tp1: np.ndarray, states: np.ndarray) -> np.ndarray:
    '''
    Format as a single vector
    '''
    T = len(taken_actions)
    # pre-allocate space for S0, A0, R1, S1, A1, R2, S2, A2, R3, ...
    trajectory = np.zeros(3*T)
    for t in range(T):
        trajectory[3*t] = states[t]
        trajectory[3*t+1] = taken_actions[t]
        trajectory[3*t+2] = rewards_tp1[t]
    return trajectory


def print_trajectory(trajectory: np.ndarray):
    '''
    Indices must allow to interpret the trajectory according
    to convention in Sutton & Barto's book, where after the
    action at time t, we obtain the reward of time t+1.'''
    T = len(trajectory) // 3
    t = 0
    print("time=" + str(t) + "  R in undefined" + "  S" + str(t) + "=" +
          str(int(trajectory[3*t])) + "  A" + str(t) + "=" + str(int(trajectory[3*t+1])))
    for t in range(1, T):
        print("time=" + str(t) +
              "  R" + str(t) + "=" + str(trajectory[3*(t-1)+2]) +
              "  S" + str(t) + "=" + str(int(trajectory[3*t])) + "  A" + str(
            t) + "=" + str(int(trajectory[3*t+1])))
    print("time=" + str(T) + "  Last reward R" +
          str(T) + "=" + str(trajectory[-1]))


def hyperparameter_grid_search(env: gym.Env):
    '''
    Grid search over alphas and epsilons
    '''
    alphas = (0.1, 0.5, 0.99)
    epsilons = (0.1, 0.001)
    print("Started grid search over alphas and epsilons")
    for a in alphas:
        for e in epsilons:
            env.reset()
            action_values, rewardsQLearning = q_learning_several_episodes(
                env, num_runs=1, episodes_per_run=2, stepSizeAlpha=a, explorationProbEpsilon=e,
                max_num_time_steps_per_episode=5, verbosity=0)
            print("Reward =", np.mean(rewardsQLearning),
                  " for grid search for alpha=", a, 'epsilon=', e)
            # fileName = 'smooth_q_eps' + str(e) + '_alpha' + str(a) + '.txt'
            # sys.stdout = open(fileName, 'w')

def theoretical_value_function(P, R, gama = 0.9):
    # Função generalizada
    s = P.shape[0]
    a = P.shape[1]
    def fun(k):
        return np.array([
            k[i] - max([
                sum([P[i, j, w] * (R[i, j, w] + gama * k[w]) for w in range(s)])
                for j in range(a)
            ])
            for i in range(s)
        ])
    print("made the system")

    # Solução inicial
    initial_guess = np.ones(s)

    # Opções de otimização (tolerâncias)
    options = {'xtol': 1e-10, 'ftol': 1e-10}
    print("solving the system")
    # Solução
    sol = fsolve(fun, initial_guess, xtol=options['xtol'])
    return sol

def  ValueFunctionFromQtable(Q):
    vF = []
    for i in range(len(Q)):
        vF.append(max(Q[i]))
    return vF

def TODO_estimate_model_probabilities(env: gym.Env):
    '''
    # AK-TODO
    Estimate dynamics using Monte Carlo.
    '''
    pass

def compare_qlearning_VI(env, vi_agent, qlearning_agent,
                                           max_num_time_steps_per_episode=100,
                                           num_episodes=10,
                                           explorationProbEpsilon=0.2,
                                           output_files_prefix=None):
    # find and use optimum policy
    
    action_values = vi_agent.Q_table
    stopping_criteria = vi_agent.hist
    iteration = stopping_criteria.shape[0]
    stopping_criterion = stopping_criteria[-1]
    print("\nMethod compute_optimal_action_values() converged in",
          iteration, "iterations with stopping criterion=", stopping_criterion)
    optimum_policy = vi_agent.get_policy()
    optimal_rewards = run_several_episodes(env, optimum_policy,
                                                max_num_time_steps_per_episode=max_num_time_steps_per_episode,
                                                num_episodes=num_episodes)
    average_reward = np.mean(optimal_rewards)
    stddev_reward = np.std(optimal_rewards)
    print('\nUsing optimum policy, average reward=',
          average_reward, ' standard deviation=', stddev_reward)

    # learn a policy with Q-learning. Use a single run.
     
    stateActionValues = qlearning_agent.Q_table
    rewardsQLearning = qlearning_agent.hist
    print('stateActionValues:', stateActionValues)
    print('rewardsQLearning:', rewardsQLearning)

    # print('Using Q-learning, total reward over training=',np.sum(rewardsQLearning))
    qlearning_policy = convert_action_values_into_policy(
        stateActionValues)
    qlearning_rewards = run_several_episodes(env, qlearning_policy,
                                                  max_num_time_steps_per_episode=max_num_time_steps_per_episode,
                                                  num_episodes=num_episodes)
    average_reward = np.mean(qlearning_rewards)
    stddev_reward = np.std(qlearning_rewards)
    print('\nUsing Q-learning policy, average reward=',
          average_reward, ' standard deviation=', stddev_reward)

    print('Check the Q-learning policy:')
    env.pretty_print_policy(qlearning_policy)

    if not output_files_prefix == None:
        with open(output_files_prefix + '_optimal.txt', 'w') as f:
            f.write(str(optimal_rewards) + "\n")

        with open(output_files_prefix + '_qlearning.txt', 'w') as f:
            f.write(str(qlearning_rewards) + "\n")

        print("Wrote files", output_files_prefix + "_optimal.txt",
              "and", output_files_prefix + "_qlearning.txt.")


if __name__ == '__main__':

    # test_dealing_with_sparsity()

    values = np.array([[3, 5, -4, 2], [10, 10, 0, -20]])
    policy = convert_action_values_into_policy(values)
    print("values =", values)
    print("policy =", policy)

    trajectory = np.array([1, 2, 3, 4, 5, 6])
    print("trajectory as an array=", trajectory)
    print("formatted trajectory:")
    print_trajectory(trajectory)
    # test_with_sparse_NextStateProbabilitiesEnv()
    # test_with_NextStateProbabilitiesEnv()

