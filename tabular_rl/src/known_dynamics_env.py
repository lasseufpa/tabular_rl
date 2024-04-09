'''
Gym (gymnasium) environment for a canonical MDP with known dynamics.
The dynamics is specified by the next state probability distributions 
p(s'/s,a) and a deterministic rewards table r(s,a,s').
Because we assume p(s'/s,a) and r(s,a,s') are the "correct" (represent
the actual world), one can calculate the optimum solution using fast
iterative methods called value iteration.

This code uses four arrays:
a) nextStateProbability: next state probabilities
b) rewardsTable: table with reward values

The other two arrays we use can be created on-the-fly.
The first one is needed for creating trajectories:
c) possible_actions_per_state
and the second one is used only for "training", in methods such as compute_optimal_action_values:
d) valid_next_states
These two arrays speed up computations but consume RAM memory. Maybe we
will have to test things without them.

This version assumes the rewards are associated to the triple
(s,a,s') (correspond to a Mealy machine),
It also assumes full (not sparse) matrices.

The initial state determined by reset() is random: any possible state.

This implementation is very general in the sense that it represents internally
rewards, actions and states as natural numbers. For instance, if a grid-world
has actions "left", "right", "up", "down", they must be mapped to integers such
as 0, 1, 2 and 3.

The finite MDP class is constructed based on an environment, which is an OpenAI's
gymnasium.Env with spaces.Discrete() for both states (called observations in gym)
and actions.

If the environment wants to have knowledge about the labels associated
to the natural numbers used with/in this MDP class, use
@VerboseKnownDynamicsEnv

@TODO:
 - support Moore (rewards associated to states) instead of only Mealy (rewards associated to transitions) (see https://www.youtube.com/watch?v=YiQxeuB56i0)
 - ??? use a single 5-dimension array instead of two matrices (see Example Exercise 3.4 "Give a table analogous to that in Example 3.3)
   To decide about it, I think we can simply check the size of the 2 matrices against one.
 - move code that generates trajectories to a class of agent, given that an environment do not create actions
 - support sparse matrices: e.g. assume the next state probability is sparse or not
 - use objects already available on gym or gymnasium.Env such as env.unwrapped.get_action_meanings()
 - Does the env as an gymnasium.Env have overhead or gets slower?
 - While the class that finds the optimum policy has some support to sparse matrices, this class here does not yet
 - Look at the guidelines to create a gym env:
https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
and
https://gymnasium.farama.org/main/tutorials/environment_creation/
'''
from __future__ import absolute_import, division, print_function

import numpy as np
import random
from random import choices, randint
import gym
from gym import spaces

class KnownDynamicsEnv(gym.Env):
    def __init__(self, nextStateProbability, rewardsTable):
        super(KnownDynamicsEnv, self).__init__()
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        self.nextStateProbability = nextStateProbability
        self.rewardsTable = rewardsTable  # expected rewards
        self.truncated = False
        self.current_observation_or_state = np.array(0, dtype=np.int32)

        # (S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
        self.S = nextStateProbability.shape[0]  # number of states
        self.A = nextStateProbability.shape[1]  # number of actions

        self.possible_states = np.arange(self.S)

        # initialize possible states and actions
        # we need to indicate only valid actions for each state
        # create a list of lists, that indicates for each state, the list of allowed actions
        self.possible_actions_per_state = self.get_valid_next_actions()

        # similar for states
        self.valid_next_states = self.get_valid_next_states()
        Low = np.array([0], dtype=np.int32)
        High = np.array([self.S-1], dtype=np.int32)
        self.action_space = spaces.Discrete(self.A)
        # states are called observations in gym
        self.observation_space = spaces.Box(Low,High, dtype=np.int32)
        self.currentIteration = 0
        self.reset()

    def get_valid_next_actions(self):
        '''
        Pre-compute valid next actions.
        Recall that the 3-D array nextStateProbability indicates p(s'/s,a),
        and has dimension S x A x S.
        This matrix specifies that actions are invalid in a given state by
        having only zeros for a given pair (s,a). For instance, assuming S=2 states
        an A=3 actions, the matrix has an invalid action for pair (s=0, a=2):
        nextStateProbability= [[[0.1 0.9]
            [1.  0. ]
            [0.  0. ]]   <=== This indicates that action a=2 is invalid while in state 0.
            [[0.6 0.4]
            [0.  1. ]
            [1.  0. ]]]
        '''
        possible_actions_per_state = list()
        for s in range(self.S):
            possible_actions_per_state.append(list())
            for a in range(self.A):
                # check if array for pair (s,a) has only zeros:
                sum_for_s_a_pair = np.sum(self.nextStateProbability[s, a])
                if sum_for_s_a_pair > 0:  # valid only if sum is larger than 0
                    possible_actions_per_state[s].append(a)
        return possible_actions_per_state

    def get_valid_next_states(self):
        '''
        Pre-compute valid next states.
        See @get_valid_next_actions
        '''
        # creates a list of lists
        valid_next_states = list()
        for s in range(self.S):
            valid_next_states.append(list())
            for a in range(self.A):
                for nexts in range(self.S):
                    p = self.nextStateProbability[s, a, nexts]
                    if p != 0:
                        # here, we temporarilly allow to have duplicated entries
                        valid_next_states[s].append(nexts)
        # now we eliminate eventual duplicated entries
        for s in range(self.S):
            # convert to set
            valid_next_states[s] = set(valid_next_states[s])
            # convert back to list again
            valid_next_states[s] = list(valid_next_states[s])
        return valid_next_states

    def step(self, action: int):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : array of topN integers
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            info (dict) :
        """
        s = self.get_state()

        # check if the chosen action is within the set of valid actions for that state
        valid_actions = self.possible_actions_per_state[s]
        if not (action in valid_actions):
            raise Exception("Action " + str(action) +
                            " is not in valid actions list: " + str(valid_actions))

        # find next state
        weights = self.nextStateProbability[s, action]
        nexts = choices(self.possible_states, weights, k=1)[0]

        # find reward value
        reward = self.rewardsTable[s, action, nexts]

        gameOver = False  # this is a continuing FMDP that never ends

        # history version with actions and states, not their indices
        # history = {"time": self.currentIteration, "state_t": self.stateListGivenIndex[s], "action_t": self.actionListGivenIndex[action],
        #           "reward_tp1": reward, "state_tp1": self.stateListGivenIndex[nexts]}
        history = {"time": self.currentIteration, "state_t": s, "action_t": action,
                   "reward_tp1": reward, "state_tp1": nexts}
        
        # update for next iteration
        self.currentIteration += 1  # update counter
        if self.currentIteration > 100:
            self.truncated = True
        self.current_observation_or_state = np.array([nexts], dtype=np.int32)

        # state is called observation in gym API
        ob = np.array([nexts], dtype=np.int32)
        return ob, float(reward), gameOver, history

    def postprocessing_MDP_step(env, history: dict, printPostProcessingInfo: bool):
        '''This method can be overriden by subclass and process history'''
        pass  # no need to do anything here

    def get_state(self):
        """Get the current observation."""
        return self.current_observation_or_state

    def reset(self, seed = None):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        aux = {}
        if seed != None:
            random.seed(seed)
        self.currentIteration = 0
        # note there are several versions of randint!
        self.current_observation_or_state = randint(0, self.S - 1)

        return np.array([self.current_observation_or_state], dtype=np.int32), aux

    def get_uniform_policy_for_known_dynamics(self):
        '''
        Takes in account the dynamics of the defined environment
        when defining actions that can be performed at each state.
        See @get_uniform_policy_for_fully_connected for
        an alternative that does not have restriction.
        '''
        policy = np.zeros((self.S, self.A))
        for s in range(self.S):
            # possible_actions_per_state is a list of lists, that indicates for each state, the list of allowed actions
            valid_actions = self.possible_actions_per_state[s]
            # no problem if denominator is zero
            uniform_probability = 1.0 / len(valid_actions)
            for a in range(len(valid_actions)):
                policy[s, a] = uniform_probability
        return policy

    def pretty_print_policy(self, policy: np.ndarray):
        '''
        Print policy.
        '''
        for s in range(self.S):
            currentState = s
            print('\ns' + str(s) + '=' + str(currentState))
            first_action = True
            for a in range(self.A):
                if policy[s, a] == 0:
                    continue
                currentAction = a
                if first_action:
                    print(' | a' + str(a) + '=' + str(currentAction), end='')
                    first_action = False  # disable this way of printing
                else:
                    print(' or a' + str(a) + '=' + str(currentAction), end='')
        print("")


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

    env = KnownDynamicsEnv(nextStateProbability, rewardsTable)
    
    #x = suite_gym.load("test")
    
