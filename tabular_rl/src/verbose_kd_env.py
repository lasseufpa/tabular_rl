'''
Verbose canonical known-dynamics environment.
It adds a list and a dictionary for actions and states:
            self.actionDictionaryGetIndex
            self.actionListGivenIndex
            self.stateDictionaryGetIndex
            self.stateListGivenIndex
such that the output to the user can be more verbose, indicating
in the grid-world example, the labels "left", "right", "up", "down", 
for instance. In the verbose case, the environment provides
the label information via the lists: stateListGivenIndex and actionListGivenIndex.
'''
import numpy as np
from random import choices, randint
import gym
from gym import spaces
from tabular_rl.src.known_dynamics_env import KnownDynamicsEnv


class VerboseKnownDynamicsEnv(KnownDynamicsEnv):
    def __init__(self, nextStateProbability, rewardsTable,
                 states_info=None, actions_info=None):
        super(VerboseKnownDynamicsEnv, self).__init__(
            nextStateProbability, rewardsTable)
        # nextStateProbability, rewardsTable)
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        if actions_info == None:
            # initialize with default names and structures
            self.actionDictionaryGetIndex, self.actionListGivenIndex = createDefaultDataStructures(
                self.A, "A")
        else:
            self.actionDictionaryGetIndex = actions_info[0]
            self.actionListGivenIndex = actions_info[1]

        if states_info == None:
            # initialize with default names and structures
            self.stateDictionaryGetIndex, self.stateListGivenIndex = createDefaultDataStructures(
                self.S, "S")
        else:
            self.stateDictionaryGetIndex = states_info[0]
            self.stateListGivenIndex = states_info[1]

    def step(self, action: int):
        ob, reward, gameOver, history = super().step(action)
        # convert history to a more pleasant version
        s = history["state_t"]
        action = history["action_t"]
        nexts = history["state_tp1"]
        # history version with actions and states, not their indices
        history = {"time": self.currentIteration, "state_t": self.stateListGivenIndex[s], "action_t": self.actionListGivenIndex[action],
                   "reward_tp1": reward, "state_tp1": self.stateListGivenIndex[nexts]}
        return ob, reward, gameOver, history

    def pretty_print_policy(self, policy: np.ndarray):
        '''
        Print policy.
        '''
        for s in range(self.S):
            currentState = self.stateListGivenIndex[s]
            print('\ns' + str(s) + '=' + str(currentState))
            first_action = True
            for a in range(self.A):
                if policy[s, a] == 0:
                    continue
                currentAction = self.actionListGivenIndex[a]
                if first_action:
                    print(' | a' + str(a) + '=' + str(currentAction), end='')
                    first_action = False  # disable this way of printing
                else:
                    print(' or a' + str(a) + '=' + str(currentAction), end='')
        print("")


def createDefaultDataStructures(num, prefix) -> tuple[dict, list]:
    '''Create default data structures for actions and states.
    '''
    possibleActions = list()
    for uniqueIndex in range(num):
        possibleActions.append(prefix + str(uniqueIndex))
    dictionaryGetIndex = dict()
    listGivenIndex = list()
    for uniqueIndex in range(num):
        dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
        listGivenIndex.append(possibleActions[uniqueIndex])
    return dictionaryGetIndex, listGivenIndex


if __name__ == '__main__':
    print("Main:")
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

    env = VerboseKnownDynamicsEnv(nextStateProbability, rewardsTable)
    print(env.step(1))
