import numpy as np
from src.knowm_dynamics_env import KnownDynamicsEnv
from src import finite_mdp_utils as fmdp


class SimpleKnownDynamicsEnv(KnownDynamicsEnv):
    def __init__(self):
        self.__version__ = "0.1.0"
        # make it a "left-right" Markov process without skips:
        # the state index cannot decrease over time nor skip
        # over the next state
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
        KnownDynamicsEnv.__init__(self, nextStateProbability, rewardsTable)

    def reset(self) -> int:
        # make sure initial state is 0
        super().reset()
        self.current_observation_or_state = 0
        return self.current_observation_or_state


if __name__ == "__main__":
    env = SimpleKnownDynamicsEnv()
    num_steps = 10
    uniform_policy = fmdp.get_uniform_policy_for_fully_connected(env.S, env.A)
    taken_actions, rewards_tp1, states = fmdp.generate_trajectory(
        env, uniform_policy, num_steps)
    trajectory = fmdp.format_trajectory_as_single_array(
        taken_actions, rewards_tp1, states)
    print("Complete trajectory vector:")
    print(trajectory)
    print("Interpret trajectory with print_trajectory() method:")
    fmdp.print_trajectory(trajectory)
