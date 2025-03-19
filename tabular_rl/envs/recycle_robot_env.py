import numpy as np

from tabular_rl.src.knowm_dynamics_env import VerboseKnownDynamicsEnv
from tabular_rl import finite_mdp_utils as fmdp
import tabular_rl.src.optimum_values as optimum
from tabular_rl.src.class_vi import VI_agent as VI

class RecycleRobotEnv(VerboseKnownDynamicsEnv):
    def __init__(self):
        self.__version__ = "0.1.0"
        # create the environment
        nextStateProbability, rewardsTable = self.recycle_robot_matrices()

        # create data structures (dic and list) to map names into indices for actions
        actionDictionaryGetIndex, actionListGivenIndex, actcomp = self.createActionsDataStructures()
        actions_info = [actionDictionaryGetIndex, actionListGivenIndex, actcomp]

        # create data structures (dic and list) to map names into indices for states
        stateDictionaryGetIndex, stateListGivenIndex, statecomp = self.createStatesDataStructures()
        states_info = [stateDictionaryGetIndex, stateListGivenIndex, statecomp]

        # superclass constructor
        VerboseKnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable,
                                         actions_info=actions_info, states_info=states_info)

    def recycle_robot_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        S = 2  # number of states
        A = 3  # number of actions
        alpha = 0.1
        beta = 0.4
        rsearch = -1
        rwait = -2
        # states
        high = 0
        low = 1
        # actions
        search = 0
        wait = 1
        recharge = 2

        # nextStateProbability Table
        nextStateProbability = np.zeros([S, A, S])
        nextStateProbability[high, search, high] = alpha
        nextStateProbability[high, search, low] = 1-alpha
        nextStateProbability[low, search, high] = 1 - beta
        nextStateProbability[low, search, low] = beta
        nextStateProbability[high, wait, high] = 1
        nextStateProbability[high, wait, low] = 0
        nextStateProbability[low, wait, high] = 0
        nextStateProbability[low, wait, low] = 1
        nextStateProbability[low, recharge, high] = 1
        nextStateProbability[low, recharge, low] = 0
        nextStateProbability[0,2, 1] = 0.5
        nextStateProbability[0,2, 0] = 0.5

        rewardsTable = np.zeros([S, A, S])  # these rewards can be negative
        rewardsTable[high, search, high] = rsearch
        rewardsTable[high, search, low] = rsearch
        rewardsTable[low, search, high] = -3
        rewardsTable[low, search, low] = rsearch
        rewardsTable[high, wait, high] = rwait
        rewardsTable[low, wait, low] = rwait

        return nextStateProbability, rewardsTable

    def createActionsDataStructures(self):
        '''
        Nice names for the actions.
        # actions
        search = 0
        wait = 1
        recharge = 2
        '''
        possibleActions = ['search', 'wait', 'recharge']
        dictionaryGetIndex = dict()
        listGivenIndex = list()
        for uniqueIndex in range(len(possibleActions)):
            dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
            listGivenIndex.append(possibleActions[uniqueIndex])
        
        return dictionaryGetIndex, listGivenIndex, possibleActions

    def createStatesDataStructures(self):
        '''
        Nice names for states
        # states
        high = 0
        low = 1
        '''
        possibleStates = ['high', 'low']
        dictionaryGetIndex = dict()
        listGivenIndex = list()
        for uniqueIndex in range(len(possibleStates)):
            dictionaryGetIndex[possibleStates[uniqueIndex]] = uniqueIndex
            listGivenIndex.append(possibleStates[uniqueIndex])
        return dictionaryGetIndex, listGivenIndex, possibleStates


if __name__ == "__main__":
    env = RecycleRobotEnv()
    print("About environment:")
    print("Num of states =", env.S)
    print("Num of actions =", env.A)
    #print("env.possible_actions_per_state =", env.possible_actions_per_state)
    print("env.nextStateProbability =", env.nextStateProbability)
    print("env.rewardsTable =", env.rewardsTable)

    uniform_policy = fmdp.get_uniform_policy_for_fully_connected(env.S, env.A)
    num_steps = 10
    taken_actions, rewards_tp1, states = fmdp.generate_trajectory(
        env, uniform_policy, num_steps)
    trajectory = fmdp.format_trajectory_as_single_array(
        taken_actions, rewards_tp1, states)
    print("Complete trajectory vector:")
    print(trajectory)
    print("Interpret trajectory with print_trajectory() method:")
    fmdp.print_trajectory(trajectory)

    tolerance = 0
    VI_agent = VI(env, tolerance=tolerance)
    state_values = VI_agent.get_vf()
    iteration = len(VI_agent.hist)
    print('Optimal solutions to the recycle robot example')
    print('Number of iterations = ', iteration)
    print('State values:')
    print(state_values)
    print(np.round(state_values, 4))
