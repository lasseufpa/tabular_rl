from __future__ import absolute_import, division, print_function

import numpy as np
import gym
from tabular_rl import simple_known_dynamics as skd
from tabular_rl import finite_mdp_utils as fmdp
from tabular_rl import known_dynamics_env as kde
from tabular_rl import optimum_values as optimum
import base64
#import imageio
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
#import pyvirtualdisplay
import reverb

import tensorflow as tf
from tf_agents.policies import policy_saver
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.environments import utils
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

kd = kde.KnownDynamicsEnv
nextStateProbability = np.array([[[0.5, 0.5, 0],
                                   [0.9, 0.1, 0]],
                                  [[0, 0.5, 0.5],
                                   [0, 0.2, 0.8]],
                                  [[0, 0.5, 0.5],
                                   [0.4, 0.3, 0.3]]])
rewardsTable = np.array([[[-3, 0, 0],
                         [2, 5, 5]],
                        [[4, 5, 0],
                         [2, 2, 6]],
                        [[8, 2, 1],
                         [11, 80, 3]]])

env = kd(nextStateProbability, rewardsTable)
env = suite_gym.wrap_env(env)
#env_name = 'CartPole-v0'
#env = suite_gym.load(env_name)


train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)


fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

time_step = eval_env.reset()
#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    while not time_step.is_last():
      action_step = policy.action(time_step)
      
      time_step = environment.step(action_step.action)
      #print(time_step.observation.__array__(np.int32), action_step.action.__array__(np.int32))
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())






example_environment = tf_py_environment.TFPyEnvironment(env)
     

time_step = example_environment.reset()
avg_return = compute_avg_return(example_environment, random_policy, 5)

policydir = "/home/caio/Documents/gitProjects/test/tabular_rl/tfPolicy/policyDqn"
saved_policy = tf.saved_model.load("/home/caio/Documents/gitProjects/test/tabular_rl/tfPolicy/policyDqn")
avg_return = compute_avg_return(example_environment, saved_policy, 5)


time_step = example_environment.reset()

#TimeStep(step_type=TensorSpec(shape=(None,), dtype=tf.int32, name='step_type'),
#         reward=TensorSpec(shape=(None,), dtype=tf.float32, name='reward'),
#         discount=TensorSpec(shape=(None,), dtype=tf.float32, name='discount'),
#         observation=TensorSpec(shape=(None, 1), dtype=tf.int32, name='observation'))


input_tensor_spec = time_step

time_step_spec = ts.time_step_spec(input_tensor_spec)


observation = tf.constant([[2]], shape=input_tensor_spec.observation.shape, dtype=np.int32)
step_type = tf.zeros(input_tensor_spec.step_type.shape, dtype=np.int32)
reward = tf.zeros(input_tensor_spec.reward.shape, dtype=np.float32)
discount = tf.ones(input_tensor_spec.discount.shape, dtype=np.float32)

time_stepe = ts.TimeStep(step_type, reward, discount, observation)


#print(time_stepe,"\n", time_step)

action_step = saved_policy.distribution(time_stepe).action
print(action_step.prob([1]))