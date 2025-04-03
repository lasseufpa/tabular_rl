import numpy as np
import pickle
from typing import Tuple, List
from src.finite_mdp_utils import ValueFunctionFromQtable as qt2vf
from src.finite_mdp_utils import convert_action_values_into_policy as getPol


class VI_agent():
    def __init__(self, env, gamma=0.9, tolerance=1e-4, debug=False):
        r = env.get_dynamics()[1]
        
        if type(r) == type(None):
            Q_table, hist = valueIteration_dictEnv(env, gamma, tolerance, debug)
        else:
            Q_table, hist = valueInteration_listEnv(env, gamma, tolerance, debug)

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

###########for dict#####
def valueIteration_dictEnv(env,
                       discountGamma: float = 0.9,
                       tolerance: float = 1e-4,
                       debug = False)-> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Value Iteration to compute the optimal action-value function.

    :param env: Environment with states, actions, and transition probabilities.
    :param discountGamma: Discount factor for future rewards.
    :param tolerance: Convergence threshold for stopping criteria.
    :return: Tuple containing the optimal action-value function and convergence metrics.
    """
    S, A = env.S, env.A
    new_action_values = np.zeros((S, A))
    stopping_criteria_per_iteration = []
    maxdif = float("inf")
    iteration = 1

    if isinstance(env.nextStateProbability, str):
        dynamic_files_inf = env.get_identify_inf()["file_inf"]
        
        transition_func = lambda: _load_dynamic_files(env.nextStateProbability, dynamic_files_inf)
    
    else:
        transition_func = lambda: iter(env.nextStateProbability.items())

    while maxdif > tolerance:
        if debug:  # debug
            print('new state values=', new_action_values)
            print('it=', iteration, 'improvement = ', maxdif)
        
        maxdif = _iterate_value_function(
            transition_func,
            new_action_values,
            discountGamma,
            A,
            stopping_criteria_per_iteration
        )
        
        iteration += 1
        print(iteration)




    return new_action_values, np.array(stopping_criteria_per_iteration)

def _load_dynamic_files(base_path: str, file_info: dict) :
    transitions = {}
    """Load dynamic files and yield their transitions."""
    for file, (start, end) in file_info.items():
        path = f"{base_path}test{file}.pkl"
        with open(path, 'rb') as f:
            transitions.clear()
            transitions = pickle.load(f)
        yield from transitions.items()

def _iterate_value_function(transition_func, action_values, discountGamma, A, stopping_criteria):
    """
    Perform a single iteration of the value function update.

    :param transition_func: A function to retrieve transitions.
    :param action_values: The current action-value function.
    :param discountGamma: Discount factor for future rewards.
    :param A: Number of actions.
    :param stopping_criteria: List to track convergence metrics.
    :return: Maximum difference in Q-values for this iteration.
    """
    maxdif = 0
    max_q = float("-inf")
    max_q_ant = float("-inf")
    value = 0
    prev_state, prev_action = 0, 0

    for (state_action, (prob, reward)) in transition_func():
        
        state, action = divmod(state_action[0], A)
        best_next_action = max(action_values[state_action[1]])  # Best Q-value for the next state

        # If transitioning to a new state or action, update Q-values
        if state != prev_state or action != prev_action:
            if state != prev_state:
                dif = abs(max_q_ant - max_q)
                maxdif = max(maxdif, dif)
                max_q, max_q_ant = float("-inf"), float("-inf")

            max_q = max(max_q, value)
            max_q_ant = max(max_q_ant, action_values[prev_state, prev_action])
            action_values[prev_state, prev_action] = value
            value = 0

        value += prob * (reward + discountGamma * best_next_action)
        prev_state, prev_action = state, action

    # Final update for the last transition
    dif = abs(max_q_ant - max_q)
    maxdif = max(maxdif, dif)
    action_values[prev_state, prev_action] = value
    stopping_criteria.append(maxdif)

    return maxdif

###########for list###############
def valueInteration_listEnv(env,
                            discountGamma=0.9,
                            tolerance=1e-4,
                            debug = False) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the action value function q_*(s,a) via the Bellman optimality
    equation for action values.
    This version assumes sparsity of the nextStateProbability array. 
    It only goes over the next states that are feasible.
    In [Sutton, 2020] the main result of this method in called "the optimal
    action-value function" and it is defined in Eq. (3.16) in page 63 and
    Eq. (3.20) in page 64.'''

    #assert isinstance(env, KnownDynamicsEnv)
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
                    
                    #print(nexts)
                    # p(s'|s,a) = p(nexts|s,a)
                    p = env.nextStateProbability[s, a, nexts]
                    # r(s,a,s') = r(s,a,nexts)
                    r = env.rewardsTable[s, a, nexts]
                    best_a = -np.inf
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
        if debug:  # debug
            print('state values=', action_values)
            print('new state values=', new_action_values)
            print('it=', iteration, 'improvement = ', improvement)
        stopping_criteria.append(improvement)
        if improvement <= tolerance:
            action_values = new_action_values.copy()
            break

        action_values = new_action_values.copy()
        iteration += 1
        print(iteration)

    return action_values, np.array(stopping_criteria)
