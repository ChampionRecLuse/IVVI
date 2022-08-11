# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation in the offline mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

from absl import logging


import gin
import numpy as np
import tensorflow.compat.v1 as tf
from brac import dataset
from brac import policies
from brac import train_eval_utils
from brac import utils
import matplotlib.pyplot as plt

@gin.configurable
def train_eval_offline(
    # Basic args.
    log_dir,
    data_file,
    agent_module,
    env_name='Instrumental_Variable_Value_Iteration',
    n_train=int(1e6),
    shuffle_steps=0,
    seed=0,
    use_seed_for_data=False,
    # Train and eval args.
    eval_target=1000,
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(2e4),
    eval_freq=5000,
    n_eval_episodes=1,
    # Agent args.
    model_params=(((200, 200),), 2),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    replay_buffer_size=int(1e6),
    initial_explore_steps=10000,
    need_eval=True,
    ):

  """Training a policy with a fixed dataset."""
  # Create tf_env to get specs.
  tf_env = train_eval_utils.env_factory(env_name)
  tf_env_test = train_eval_utils.env_factory(env_name)
  observation_spec = tf_env.observation_spec()
  action_spec = tf_env.action_spec()

  # Initialize dataset.
  with tf.device('/cpu:0'):
    train_data = dataset.Dataset(
        observation_spec,
        action_spec,
        replay_buffer_size,
        circular=True,
        )
  data_ckpt = tf.train.Checkpoint(data=train_data)
  data_ckpt_name = os.path.join(log_dir, 'replay')

  time_st_total = time.time()
  time_st = time.time()
  timed_at_step = 0

  # Collect data from random policy.
  explore_policy = policies.ContinuousRandomPolicy(action_spec)
  steps_collected = 0
  log_freq = 5000
  logging.info('Collecting data ...')
  collector = train_eval_utils.DataCollector(tf_env, explore_policy, train_data)
  while steps_collected < initial_explore_steps:
    count = collector.collect_transition()
    steps_collected += count
    if (steps_collected % log_freq == 0
        or steps_collected == initial_explore_steps) and count > 0:
      steps_per_sec = ((steps_collected - timed_at_step)
                       / (time.time() - time_st))
      timed_at_step = steps_collected
      time_st = time.time()
      logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                   initial_explore_steps, steps_per_sec)

  # Create agent.
  agent_flags = utils.Flags(
      observation_spec=observation_spec,
      action_spec=action_spec,
      model_params=model_params,
      optimizers=optimizers,
      batch_size=batch_size,
      weight_decays=weight_decays,
      update_freq=update_freq,
      update_rate=update_rate,
      discount=discount,
      train_data=train_data)
  agent_args = agent_module.Config(agent_flags).agent_args
  agent = agent_module.Agent(**vars(agent_args))
  agent_ckpt_name = os.path.join(log_dir, 'agent_final')

  # Restore agent from checkpoint if there exists one.
  if tf.io.gfile.exists('{}.index'.format(agent_ckpt_name)):
    print('GOT IT')
    logging.info('Checkpoint found at %s.', agent_ckpt_name)
    agent.restore(agent_ckpt_name)

  # train_summary_dir = os.path.join(log_dir, 'train')
  # eval_summary_dir = os.path.join(log_dir, 'eval')
  # eval_result, eval_infos = train_eval_utils.eval_policies(
  #     tf_env_test, agent.test_policies, n_eval_episodes)
  # results = []
  ACTION = []
  STATE = []
  for name, policy in agent.test_policies.items():
      for _ in range(1):
        time_step = tf_env_test.reset()
        STATE.append(time_step.observation.numpy())
        total_rewards = 0.0
        while not time_step.is_last().numpy()[0]:
          action = policy(time_step.observation)[0]
          ACTION.append(action.numpy())
          time_step = tf_env_test.step(action)
          STATE.append(time_step.observation.numpy())
          total_rewards += time_step.reward
        print(total_rewards)
  print(ACTION)
  print(STATE)

  ACTION = np.array(ACTION)
  STATE = np.array(STATE)
  A = []
  S = []
  ACTION = np.reshape(ACTION,1000)
  STATE = np.reshape(STATE,1001)
  print(np.shape(ACTION))
  x = np.arange(1000)
  plt.figure(1)
  plt.plot(x,ACTION)
  plt.xlabel('Horizon')
  plt.ylabel('action')

  plt.figure(2)
  y = np.arange(1001)
  plt.plot(y,STATE)
  plt.xlabel('Horizon')
  plt.ylabel('state')

  plt.show()
