import numpy as np
from random import choices, randint, seed as Seed
import gymnasium as gym
from gymnasium import spaces
import pickle
import numpy as np
import pickle
from typing import Tuple, Dict
from scipy import sparse as sc
#from class_vi import VI_agent
#from tabular_rl import finite_mdp_utils as fmdp
#from stable_baselines3 import DQN


class KnownDynamicsEnv(gym.Env):
    def __init__(self, nextStateProbability, rewardsTable=None, NS = None, NA = None, sparcity_treshold = 0.5):
        
        type_struct = 0
        if isinstance(nextStateProbability, str) or isinstance(nextStateProbability, dict):
            struct = DictKDEnv(nextStateProbability, rewardsTable, NS, NA)

        else:

            num = np.nonzero(nextStateProbability)[0].size
            dem = nextStateProbability.size
            sparcity_rate = 1-(num/dem)
            print(sparcity_rate, sparcity_treshold)

            if sparcity_rate > sparcity_treshold:
                dynamics = prepros(nextStateProbability, rewardsTable)
                NS = nextStateProbability.shape[0]
                NA = nextStateProbability.shape[1]

                struct = DictKDEnv(dynamics, None, NS, NA)
                nextStateProbability = dynamics
                rewardsTable = None
            else:
                type_struct = 1
                struct = listKdEnv(nextStateProbability, rewardsTable, NS, NA)
        
        print(f"using {struct.__class__}")
        self.nextStateProbability = nextStateProbability
        self.rewardsTable = rewardsTable
        self.S = struct.S
        self.A = struct.A
        self.current_state = None
        self.struct = struct
        self.type_struct = type_struct
        self.currentIteration = 0

        #self.reset()
        
    def reset(self, s = None, seed = None):
        a = self.struct.reset(s, seed)
        self.current_state = a[0]
        self.currentIteration = 0
        return a

    def step(self, action):

        ret = self.struct.step(action)
        self.current_state = ret[0]
        self.currentIteration +=1
        return ret
    
    def get_obs(self):
        return self.current_state

    def get_dynamics(self):
        return self.nextStateProbability, self.rewardsTable
    
    def get_valid_next_states(self):
        if self.type_struct == 0:
            return None
        else:
            return self.struct.valid_next_states

class VerboseKnownDynamicsEnv(KnownDynamicsEnv):
    def __init__(self, nextStateProbability, rewardsTable= None, NS = None, NA = None, sparcity_treshold = 0.5,
                 states_info=None, actions_info=None):
        super(VerboseKnownDynamicsEnv, self).__init__(nextStateProbability, rewardsTable, NS, NA, sparcity_treshold)
        # nextStateProbability, rewardsTable)
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        if actions_info == None:
            # initialize with default names and structures
            self.actionDictionaryGetIndex, self.actionListGivenIndex, self.nameActionsComponents = createDefaultDataStructures(
                self.A, 1, "Ac")
        else:
            self.actionDictionaryGetIndex = actions_info[0]
            self.actionListGivenIndex = actions_info[1]
            self.nameActionsComponents = actions_info[2]

        if states_info == None:
            # initialize with default names and structures
            self.stateDictionaryGetIndex, self.stateListGivenIndex, self.nameStateComponents = createDefaultDataStructures(
                self.S,1, "Sc")
        else:
            self.stateDictionaryGetIndex = states_info[0]
            self.stateListGivenIndex = states_info[1]
            self.nameStateComponents = states_info[2]
        
        self.action_space = spaces.Discrete(self.A)

        Low = np.array(np.full(len(self.nameStateComponents),  -10))
        High = np.array(np.full(len(self.nameStateComponents), 10) )
        self.observation_space = spaces.Box(Low,High)
        
        

    def step(self, action: int):

        ob, reward, gameOver, truncated, history = super().step(action)
        verbose_next_ob = self.stateListGivenIndex[ob]
        
        # convert history to a more pleasant version
        s = int(history["state_t"])
        action = history["action_t"]
        nexts = history["state_tp1"]
        # history version with actions and states, not their indices
        history = {"time": self.currentIteration, "state_t": self.stateListGivenIndex[s], "action_t": self.actionListGivenIndex[action],
                   "reward_tp1": reward, "state_tp1": self.stateListGivenIndex[nexts]}
        
        self.verbose_obs = verbose_next_ob

        return verbose_next_ob, reward, gameOver, truncated, history
    
    def reset(self, s=None, seed=None):

        if isinstance(s, tuple):
            s = self.stateDictionaryGetIndex[s]
        state_int = super().reset(s, seed)[0]
        verbose_state = self.stateListGivenIndex[state_int]
        self.verbose_obs = verbose_state
        return self.verbose_obs, {}
    

    def get_dynamics(self):
        return super().get_dynamics() 
    
    def pretty_print_policy(self, policy: np.ndarray):
        '''
        Print policy.
        '''
        for s in range(self.S):
            currentState = self.stateListGivenIndex[s]
            initial_phrase = '\ns' + str(s) + '= '
            print(currentState[0])
            for i in range(len(self.nameStateComponents)):
                initial_phrase +=  self.nameStateComponents[i] + ": "+ str(currentState[i]) +", "
            print(initial_phrase)
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







class listKdEnv(gym.Env):
    def __init__(self, nextStateProbability, rewardsTable, NS= None, NA = None, nMaxStep = 100):
        #super(KnownDynamicsEnv, self).__init__()
        self.__version__ = "0.1.0"
        # print("AK Finite MDP - Version {}".format(self.__version__))
        self.nextStateProbability = nextStateProbability
        self.rewardsTable = rewardsTable  # expected rewards
        self.truncated = False
        self.current_observation_or_state = None
        self.nMaxStep = nMaxStep
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
        self.action_space = spaces.Discrete(self.A)
        # states are called observations in gym
        self.observation_space = spaces.Discrete(self.S)
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

    def get_state(self):
        """Get the current observation."""
        return self.current_observation_or_state
    
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

    def step(self, action):
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
        valid_actions = self.possible_actions_per_state[int(s)]
        if not (action in valid_actions):
            raise Exception("Action " + str(action) +
                            " is not in valid actions list: " + str(valid_actions))

        # find next state
        weights = self.nextStateProbability[int(s), action]
        nexts = choices(self.possible_states, weights, k=1)[0]

        # find reward value
        reward = self.rewardsTable[s, action, nexts]

        gameOver = False  # this is a continuing FMDP that never ends
        Truncated = False
        # history version with actions and states, not their indices
        # history = {"time": self.currentIteration, "state_t": self.stateListGivenIndex[s], "action_t": self.actionListGivenIndex[action],
        #           "reward_tp1": reward, "state_tp1": self.stateListGivenIndex[nexts]}
        history = {"time": self.currentIteration, "state_t": s, "action_t": action,
                   "reward_tp1": reward, "state_tp1": nexts}
        
        # update for next iteration
        self.currentIteration += 1  # update counter
        self.current_observation_or_state = nexts
        
        if self.currentIteration == self.nMaxStep:
            Truncated = True
            gameOver = True 
        # state is called observation in gym API
        ob = nexts
        return ob, float(reward), gameOver, Truncated, history

    def reset(self, s = None, seed = None)->int:
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        Seed(seed)
        self.currentIteration = 0
        if s != None:
            self.current_observation_or_state = s
        else:    
            # note there are several versions of randint!
            self.current_observation_or_state = randint(0, self.S - 1)
        Seed(None)
        
        return self.current_observation_or_state, {}


class DictKDEnv(gym.Env):
    def __init__(self, nextStateProbability, reward=None, NS= None, NA = None, nMaxStep = 100):
        
        indentify_text = None
        if isinstance(nextStateProbability, str):
            with open(nextStateProbability + "dynamic_inf.pkl", 'rb') as f:
                indentify_text = pickle.load(f)
            self.nextStateProbability = nextStateProbability
            NS = indentify_text["NS"]
            NA = indentify_text["NA"]

        else:
            self.nextStateProbability = nextStateProbability
            NS = NS
            NA = NA
        
        self.nMaxStep = nMaxStep
        self.indentify_text = indentify_text
        self.S = NS
        self.A = NA
        self.act_dynamic = None
        self.current_observation_or_state = 0

        # (S, A, nS) = self.nextStateProbability.shape #should not require nextStateProbability, which is often unknown
          # number of actions

        self.action_space = spaces.Discrete(self.A)
        # states are called observations in gym
        self.observation_space = spaces.Discrete(self.S)

        self.currentIteration = 0
        self.reset()



    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Execute a step in the environment based on the given action.
        
        :param action: The action taken by the agent.
        :return: A tuple containing:
            - ob: The next observation/state.
            - reward: The reward received.
            - gameOver: Whether the game has ended.
            - Truncated: Whether the episode was truncated.
            - history: A dictionary with additional information.
        """
        s = self.get_state()
        weights = np.zeros(self.S)
        
        if isinstance(self.nextStateProbability, str):  # Multiple files
            ind0 = self._get_file_and_transition(s, action, weights)
        else:  # Single file
            ind0 = s * self.A + action
            self._populate_weights(weights, self.nextStateProbability, ind0)
        
        # Select next state
        nexts = choices(range(self.S), weights, k=1)[0]
        reward = self._get_reward(ind0, nexts)
        
        # Determine terminal states and history

        history = {"time": self.currentIteration, "state_t": s, "action_t": action,
                   "reward_tp1": reward, "state_tp1": nexts}
        Truncated = False
        gameOver = False
        if self.currentIteration == 100:
            Truncated = True
            gameOver = True 
        
        # Update state and iteration
        self.currentIteration += 1
        self.current_observation_or_state = nexts
        ob = nexts
        
        return ob, float(reward), gameOver, Truncated, history
    
    def _get_file_and_transition(self, s: int, action: int, weights: np.ndarray) -> int:
        """Handles file loading and state transition for multiple files."""
        for i, (start, end) in self.indentify_text["file_inf"].items():
            if start <= s <= end:
                if not hasattr(self, "_file_cache") or self._file_cache != i:
                    dynamic_path = f"{self.nextStateProbability}test{i}.pkl"
                    with open(dynamic_path, 'rb') as f:
                        self.act_dynamic = pickle.load(f)
                    self._file_cache = i
                break
        
        ind0 = s * self.A + action
        self._populate_weights(weights, self.act_dynamic, ind0)
        return ind0
    
    def _populate_weights(self, weights: np.ndarray, probabilities: Dict, ind0: int):
        """Populates the weights array based on the transition probabilities."""
        aux = [k for k in probabilities.keys() if k[0] == ind0]
        for i in aux:
            weights[i[1]] = probabilities[i][0]
    
    def _get_reward(self, ind0: int, nexts: int) -> float:
        """Retrieves the reward value for the given transition."""
        if isinstance(self.nextStateProbability, str):
            return self.act_dynamic[(ind0, nexts)][1]
        return self.nextStateProbability[(ind0, nexts)][1]


    def get_state(self) -> int:
        """Get the current observation."""
        return self.current_observation_or_state

    def reset(self, s = None, seed = None) -> int:
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        Seed(seed)
        self.currentIteration = 0
        # note there are several versions of randint!
        if s != None:
            self.current_observation_or_state = s
        else:    
            self.current_observation_or_state = randint(0, self.S - 1)
        Seed(None)
        return self.current_observation_or_state, {}
    


######### Funcs


def createDefaultDataStructures(num, num_components, prefix) -> tuple[dict, list, list]:
    '''Create default data structures for actions and states.
    '''
    possibleActions = list()
    dictionaryGetIndex = dict()
    listGivenIndex = list()

    for uniqueIndex in range(num):
        aux = []
        for i in range(num_components):
            aux.append(uniqueIndex)
        aux = tuple(aux)
        possibleActions.append(aux)
        dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
        listGivenIndex.append(possibleActions[uniqueIndex])

        
    components_names = []
    for i in range(num_components):
        components_names.append(prefix + str(i))

    return dictionaryGetIndex, listGivenIndex, components_names


def prepros(nextStateProbability, rewardTable):

    S = nextStateProbability.shape[0]
    A = nextStateProbability.shape[1]


    nextStateProbability = nextStateProbability.reshape([A*S, S])
    nextStateProbability = sc.csr_matrix(nextStateProbability).todok()

    rewardTable = rewardTable.reshape([A*S, S])
    rewardTable =  sc.dok_matrix(rewardTable)

    auxDict = {}
    for i in nextStateProbability.keys():
        auxDict[i] = (nextStateProbability[i],rewardTable[i])

    return auxDict


if __name__ == '__main__':
    print("Main:")
    nextStateProbability = np.array([[[0.5, 0.5, 0],
                                      [0.9, 0.1, 0]],
                                     [[0, 0, 1],
                                      [0, 0.5, 0.5]],
                                     [[0, 0, 1],
                                      [0, 0, 1]]])
    rewardsTable = np.array([[[-3, 0, 0],
                              [-2, 5, 5]],
                             [[4, 5, 0],
                              [2, 2, 6]],
                             [[-8, 2, 10],
                              [11, 0, 3]]])
    ns = nextStateProbability.shape[0]
    na = nextStateProbability.shape[1]
    #dynamic = prepros(nextStateProbability, rewardsTable)
    env = VerboseKnownDynamicsEnv(nextStateProbability, rewardsTable, NS=ns, NA=na, sparcity_treshold=0.5)
    
    #print(env.get_obs())
    #vi = VI_agent(env, debug=False)
    #pol = fmdp.convert_action_values_into_policy(vi.Q_table)
    #env.pretty_print_policy(pol)
    #a = fmdp.run_several_episodes(env, pol)
    #print(a)

    #env.pretty_print_policy(pol)
    #print(env.get_state())
    #print(env.current_observation_or_state)
    #print(env.step(0))
    #x = suite_gym.load("test")
    