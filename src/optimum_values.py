import numpy as np
from scipy.optimize import fsolve
from src.knowm_dynamics_env import KnownDynamicsEnv
from src import finite_mdp_utils as fmdp


def compute_optimal_state_values(env: KnownDynamicsEnv, discountGamma=0.9, tolerance=1e-20, use_sutton_version=True, use_nonsparse_version=False) -> tuple[np.ndarray, int]:
    '''
    Compute the state value function v_*(s) via the Value Iteration algorithm,
    which is described in Eq. (4.10) of [Sutton, 2020].
    This algorithm is based on Bellman's optimality equation for state values,
    which corresponds to Eq. (4.1) and also Eq. (3.19) in page 63 of [Sutton, 2020].
    This is the public methods that calls the respective "private" method.
    '''
    assert isinstance(
        env, KnownDynamicsEnv)  # make sure env is a KnownDynamicsEnv

    if use_sutton_version:
        return __compute_optimal_state_values_sparse_as_sutton(env, discountGamma=discountGamma, tolerance=tolerance)
    elif use_nonsparse_version:
        return __compute_optimal_state_values_nonsparse(env, discountGamma=discountGamma, tolerance=tolerance)
    else:
        return __compute_optimal_state_values_for_sparse_matrices(env, discountGamma=discountGamma, tolerance=tolerance)


def __compute_optimal_state_values_nonsparse(env: KnownDynamicsEnv, discountGamma=0.9, tolerance=1e-20) -> tuple[np.ndarray, int]:
    '''
    @see compute_optimal_state_values
    This implementation uses a stopping criterion simply based on the
    provided input parameter "tolerance", which is different than the criterion
    suggested in [Sutton, 2020].
    This version does not assume in-place computation.
    This version does not assume nor explore sparsity.'''

    S = env.S  # total number of states
    A = env.A  # total number of actions

    new_state_values = np.zeros((S,))
    state_values = np.zeros((S,))
    iteration = 1
    a_candidates = np.zeros((A,))
    while True:  # until convergence
        for s in range(S):  # implement a sweep over all possible states
            # fill with zeros without creating new array
            # a_candidates[:] = 0
            a_candidates.fill(0.0)
            for a in range(A):
                value = 0
                for nexts in range(S):
                    p = env.nextStateProbability[s, a, nexts]
                    if p != 0:
                        r = env.rewardsTable[s, a, nexts]
                        value += p * (r + discountGamma * state_values[nexts])
                a_candidates[a] = value
            new_state_values[s] = np.max(a_candidates)
        improvement = np.sum(np.abs(new_state_values - state_values))
        # print('improvement =', improvement)
        if False:  # debug
            print('state values=', state_values)
            print('new state values=', new_state_values)
            print('it=', iteration, 'improvement = ', improvement)
        # I am avoiding to use np.copy() here because memory kept growing
        for i in range(S):
            state_values[i] = new_state_values[i]
        if improvement <= tolerance:
            break

        iteration += 1

    return state_values, iteration


def __compute_optimal_state_values_for_sparse_matrices(env: KnownDynamicsEnv, discountGamma=0.9, tolerance=1e-20) -> tuple[np.ndarray, int]:
    '''
    @see compute_optimal_state_values
    This version is useful when nextStateProbability is sparse. It only goes over the
    next states that are feasible.
    This version does not assume in-place computation.
    '''

    S = env.S  # total number of states

    new_state_values = np.zeros((S,))
    state_values = np.zeros((S,))
    iteration = 1
    valid_next_states = env.valid_next_states
    while True:  # until convergence
        for s in range(S):  # implement a sweep over all possible states

            # Getting the Possible actions per state
            possibleAction = env.possible_actions_per_state[s]
            # Creating a array to compute the possibles actions
            a_candidates = np.zeros(len(possibleAction))
            # print(s)
            # fill with zeros without creating new array
            # a_candidates[:] = 0
            a_candidates.fill(0.0)
            feasible_next_states = valid_next_states[s]
            num_of_feasible_next_states = len(feasible_next_states)
            for a in possibleAction:
                value = 0
                # for nexts in range(S):
                for feasible_nexts in range(num_of_feasible_next_states):
                    # print('feasible_nexts=',feasible_nexts)
                    nexts = feasible_next_states[feasible_nexts]
                    p = env.nextStateProbability[s, a, nexts]
                    r = env.rewardsTable[s, a, nexts]
                    # print('p',p,'state_values[nexts]',state_values[nexts],'r',r)
                    value += p * (r + discountGamma * state_values[nexts])
                a_candidates[a] = value
            new_state_values[s] = np.max(a_candidates)
        improvement = np.sum(np.abs(new_state_values - state_values))
        # print('improvement =', improvement)
        if False:  # debug
            print('state values=', state_values)
            print('new state values=', new_state_values)
            print('it=', iteration, 'improvement = ', improvement)
        # I am avoiding to use np.copy() here because memory kept growing
        for i in range(S):
            state_values[i] = new_state_values[i]
        if improvement <= tolerance:
            break

        iteration += 1

    return state_values, iteration


def __compute_optimal_state_values_sparse_as_sutton(env: KnownDynamicsEnv, discountGamma=0.9, tolerance=1e-20) -> tuple[np.ndarray, int]:
    '''
    @see compute_optimal_state_values
    This implementation adopts in-place calculation, as suggested
    in [Sutton, 2020] textbook, page 75, which says:
    "We usually have the in-place version in mind when we think of DP algorithms."
    Besides, this version assumes the matrices are sparse.
    '''
    S = env.S  # total number of states

    # new_state_values = np.zeros((S,))
    state_values = np.zeros((S,))
    iteration = 1
    valid_next_states = env.valid_next_states
    while True:  # until convergence
        # Initialize Delta in Value Iteration algoritm, [Sutton, 2020], page 83
        improvement_Delta = 0
        for s in range(S):  # implement a sweep over all possible states
            # Getting the Possible actions per state
            possibleAction = env.possible_actions_per_state[s]
            # Creating a array to compute the possibles actions
            a_candidates = np.zeros(len(possibleAction))
            # print(s)
            # fill with zeros without creating new array
            # a_candidates[:] = 0
            a_candidates.fill(0.0)
            feasible_next_states = valid_next_states[s]
            num_of_feasible_next_states = len(feasible_next_states)
            for a in possibleAction:
                value = 0
                # for nexts in range(S):
                for feasible_nexts in range(num_of_feasible_next_states):
                    # print('feasible_nexts=',feasible_nexts)
                    nexts = feasible_next_states[feasible_nexts]
                    p = env.nextStateProbability[s, a, nexts]
                    r = env.rewardsTable[s, a, nexts]
                    # print('p',p,'state_values[nexts]',state_values[nexts],'r',r)
                    value += p * (r + discountGamma * state_values[nexts])
                a_candidates[a] = value
            # new_state_values[s] = np.max(a_candidates)
            new_state_value = np.max(a_candidates)
            this_improvement = np.abs(new_state_value - state_values[s])
            state_values[s] = new_state_value
            # check if improvement_Delta needs to be updated
            if this_improvement > improvement_Delta:
                improvement_Delta = this_improvement
            # print('this_improvement  =', this_improvement)
        if False:  # debug
            print('state values=', state_values)
            print('it=', iteration, 'max improvement = ', improvement_Delta)
        # I am avoiding to use np.copy() here because memory kept growing
        # for i in range(S):
        #    state_values[i] = new_state_values[i]
        if improvement_Delta <= tolerance:
            break

        iteration += 1

    return state_values, iteration


def compute_optimal_action_values_nonsparse(env: KnownDynamicsEnv,
                                            discountGamma=0.9,
                                            tolerance=1e-20) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the action value function q_*(s,a) via the Bellman optimality
    equation for action values.
    This version does not assume nor explore sparsity.
    In [Sutton, 2020] the main result of this method in called "the optimal
    action-value function" and it is defined in Eq. (3.16) in page 63 and
    Eq. (3.20) in page 64.'''

    assert isinstance(env, KnownDynamicsEnv)
    S = env.S
    A = env.A
    # (S, A, nS) = env.nextStateProbability.shape
    # A = len(actionListGivenIndex)
    new_action_values = np.zeros((S, A))
    action_values = new_action_values.copy()
    iteration = 1
    stopping_criteria = list()
    while True:
        # src = new_action_values if in_place else action_values
        for s in range(S):
            for a in range(A):
                value = 0
                for nexts in range(S):
                    # p(s'|s,a) = p(nexts|s,a)
                    p = env.nextStateProbability[s, a, nexts]
                    # r(s,a,s') = r(s,a,nexts)
                    r = env.rewardsTable[s, a, nexts]
                    best_a = -np.Infinity
                    for nexta in range(A):
                        temp = action_values[nexts, nexta]
                        if temp > best_a:
                            best_a = temp
                    value += p * (r + discountGamma * best_a)
                    # value += p*(r+discount*src[nexts])
                    # print('aa', value)
                new_action_values[s, a] = value
        improvement = np.sum(np.abs(new_action_values - action_values))
        # print('improvement =', improvement)
        if False:  # debug
            print('state values=', action_values)
            print('new state values=', new_action_values)
            print('it=', iteration, 'improvement = ', improvement)
        stopping_criteria.append(improvement)
        if improvement <= tolerance:
            action_values = new_action_values.copy()
            break

        action_values = new_action_values.copy()
        iteration += 1

    return action_values, np.array(stopping_criteria)


def compute_optimal_action_values(env: KnownDynamicsEnv,
                                  discountGamma=0.9,
                                  use_nonsparse_version=False,
                                  tolerance=1e-20) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the action value function q_*(s,a) via the Bellman optimality
    equation for action values.
    This version assumes sparsity of the nextStateProbability array. 
    It only goes over the next states that are feasible.
    In [Sutton, 2020] the main result of this method in called "the optimal
    action-value function" and it is defined in Eq. (3.16) in page 63 and
    Eq. (3.20) in page 64.'''
    if use_nonsparse_version:
        return compute_optimal_action_values_nonsparse(env, discountGamma=discountGamma, tolerance=tolerance)

    assert isinstance(env, KnownDynamicsEnv)
    S = env.S
    A = env.A
    new_action_values = np.zeros((S, A))
    action_values = np.zeros((S, A))
    iteration = 1
    valid_next_states = env.valid_next_states
    stopping_criteria_per_iteration = list()
    while True:
        for s in range(S):
            feasible_next_states = valid_next_states[s]
            num_of_feasible_next_states = len(feasible_next_states)
            for a in range(A):
                value = 0
                for feasible_nexts in range(num_of_feasible_next_states):
                    nexts = feasible_next_states[feasible_nexts]
                    # p(s'|s,a) = p(nexts|s,a)
                    p = env.nextStateProbability[s, a, nexts]
                    # r(s,a,s') = r(s,a,nexts)
                    r = env.rewardsTable[s, a, nexts]
                    best_a = np.max(action_values[nexts])
                    value += p * (r + discountGamma * best_a)
                new_action_values[s, a] = value

        # check if should stop because process converged
        abs_difference_array = np.abs(new_action_values - action_values)
        stopping_criterion_option = 2  # 1 uses sum and 2 uses max
        if stopping_criterion_option == 1:
            # use the sum
            stopping_criterion = np.sum(
                abs_difference_array)/np.max(np.abs(new_action_values))
        else:
            # use the max
            stopping_criterion = np.max(
                abs_difference_array)/np.max(np.abs(new_action_values))
        stopping_criteria_per_iteration.append(stopping_criterion)
        # print('DEBUG: stopping_criterion =', stopping_criterion)
        if False:  # debug
            print('state values=', action_values)
            print('new state values=', new_action_values)
            print('it=', iteration, 'improvement = ', stopping_criterion)

        # action_values = new_action_values.copy()
        # I am avoiding to use np.copy() here because memory kept growing
        for i in range(S):
            for j in range(A):
                action_values[i, j] = new_action_values[i, j]

        if stopping_criterion <= tolerance:
            # print('DEBUG: stopping_criterion =',
            #      stopping_criterion, "and tolerance=", tolerance)
            break

        iteration += 1

    return action_values, np.array(stopping_criteria_per_iteration)


def compare_policys(env: KnownDynamicsEnv,
                                           max_num_time_steps_per_episode=100,
                                           num_episodes=10,
                                           explorationProbEpsilon=0.2,
                                           output_files_prefix=None):
    # find and use optimum policy
    env.reset()
    tolerance = 1e-10
    action_values, stopping_criteria = compute_optimal_action_values(
        env, tolerance=tolerance)
    iteration = stopping_criteria.shape[0]
    stopping_criterion = stopping_criteria[-1]
    print("\nMethod compute_optimal_action_values() converged in",
          iteration, "iterations with stopping criterion=", stopping_criterion)
    optimum_policy = fmdp.convert_action_values_into_policy(action_values)
    optimal_rewards = fmdp.run_several_episodes(env, optimum_policy,
                                                max_num_time_steps_per_episode=max_num_time_steps_per_episode,
                                                num_episodes=num_episodes)
    average_reward = np.mean(optimal_rewards)
    stddev_reward = np.std(optimal_rewards)
    print('\nUsing optimum policy, average reward=',
          average_reward, ' standard deviation=', stddev_reward)

    # learn a policy with Q-learning. Use a single run.
    stateActionValues, rewardsQLearning = fmdp.q_learning_several_episodes(
        env, num_runs=1, episodes_per_run=num_episodes,
        max_num_time_steps_per_episode=max_num_time_steps_per_episode,
        explorationProbEpsilon=explorationProbEpsilon)
    print('stateActionValues:', stateActionValues)
    print('rewardsQLearning:', rewardsQLearning)

    # print('Using Q-learning, total reward over training=',np.sum(rewardsQLearning))
    qlearning_policy = fmdp.convert_action_values_into_policy(
        stateActionValues)
    qlearning_rewards = fmdp.run_several_episodes(env, qlearning_policy,
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
    if False:
        test_dealing_with_sparsity()

    # define env
    nextStateProbability = np.array([[[0.5, 0.5, 0],
                                      [0.9, 0.1, 0]],
                                     [[0, 0.5, 0.5],
                                      [0, 0.2, 0.8]],
                                     [[0, 0, 1],
                                      [0, 0, 1]]])
    rewardsTable = np.array([[[-3, 0, 0],
                              [-2, 5, 5]],
                             [[4, 5, 0],
                              [2, 2, 6]],
                             [[-8, 2, 80],
                              [11, 0, 3]]])

    env = KnownDynamicsEnv(nextStateProbability, rewardsTable)
    discountGamma = 0.8
    tolerance = 0
    use_nonsparse_version = True
    use_sutton_version = False
    state_values, iteration = compute_optimal_state_values(
        env, discountGamma=discountGamma, tolerance=tolerance,
        use_sutton_version=True, use_nonsparse_version=False)
    print('Optimum state_values via Value Iteration=', state_values)
    print('Total number of iterations until convergence =', iteration)
