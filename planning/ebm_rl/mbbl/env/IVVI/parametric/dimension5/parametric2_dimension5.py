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

"""Training and evaluation in the online mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
from typing import Any

from absl import logging

import gin
import numpy as np
import tensorflow.compat.v1 as tf
from brac import dataset
from brac import policies
from brac import train_eval_utils
from brac import utils
from mbbl.env import env_register
import random
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from gym.spaces import box


class env(py_environment.PyEnvironment):

    def get_info(self) -> Any:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def __init__(self, env_name, rand_seed, misc_info):
        self._env_name = env_name
        self._seed = rand_seed
        self._npr = np.random.RandomState(self._seed)
        self._misc_info = misc_info
        self._env_info = env_register.get_env_info(self._env_name)
        self.dimension = 5
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-100, maximum=100, name='observation')
        self.action_space = box.Box(low=-1.0, high=1.0, shape=(self.dimension,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(self.dimension,), dtype=np.float32)
        self.P = 0.5 * np.eye(self.dimension) + np.diag(0.2 * np.ones(self.dimension-1),k=1) + np.diag(0.2 * np.ones(self.dimension-1),k=-1)
        self.Q = 0.5 * np.eye(self.dimension) + np.diag(0.1 * np.ones(self.dimension-1),k=1) + np.diag(0.1 * np.ones(self.dimension-1),k=-1)
        self.mean = np.zeros(self.dimension)
        self.identity = np.eye(self.dimension)
#         self.transition = np.array([[ 1.55238650e-03, 4.92410299e-01, 1.98844535e-01, 6.01305589e-03,
#   1.58399517e-03,-1.11683868e-02,-3.61836858e-01,-5.33642966e-02,
#   2.89601924e-02, 1.87458705e-02, 2.52975644e-02],
#  [ 1.09248132e-02, 1.92421823e-01, 4.95488699e-01, 2.01015035e-01,
#   -5.76109730e-03,-1.11972340e-02,-4.61363401e-02,-3.82780954e-01,
#   -4.51666800e-02, 3.35328057e-02, 2.90320134e-02],
#  [ 4.07093301e-03,-2.65992135e-03, 2.05003550e-01, 5.08122256e-01,
#   2.09601470e-01,-3.86499034e-03, 2.97599542e-02,-6.38588557e-02,
#   -3.68586140e-01,-4.44710006e-02, 2.72430821e-02],
#  [ 6.43951677e-03, 5.98822204e-03,-3.43427566e-04, 2.00479778e-01,
#   5.12674274e-01, 2.02008789e-01, 2.93694249e-02, 1.91691698e-02,
#   -4.46457025e-02,-3.88085188e-01,-5.84850494e-02],
#  [ 1.01703663e-02, 2.74164446e-03, 6.14620571e-03,-1.31370349e-02,
#   2.13082318e-01, 4.95260675e-01, 2.48675950e-02, 2.83267167e-02,
#   3.03529724e-02,-5.73947517e-02,-3.77492505e-01]])
   
   
        self.transition = np.array([[-6.75879150e-03, 4.95681864e-01, 2.00044127e-01,-1.06644519e-02,
  -5.62281256e-03,-7.17945210e-03,-3.73473095e-01,-4.71997026e-02,
   1.97202409e-02, 1.49050617e-02, 3.38389197e-02],
 [-1.64881265e-03, 2.07627349e-01, 4.94242257e-01, 1.90148831e-01,
  -2.57166773e-03, 9.22401554e-03,-5.31339613e-02,-3.68037930e-01,
  -5.51473554e-02, 1.22358533e-02, 1.70230878e-02],
 [ 5.87471451e-03,-1.30208990e-03, 2.00698846e-01, 5.00810302e-01,
   2.00308009e-01, 1.51970769e-03, 3.28987716e-02,-4.72018258e-02,
  -3.82301020e-01,-5.74955091e-02, 1.33777140e-02],
 [ 2.60055504e-03,-3.13829360e-03, 1.40700507e-03, 1.94418512e-01,
   5.04330127e-01, 1.94020335e-01, 3.02105048e-02, 3.41234313e-02,
  -6.35125472e-02,-3.78092321e-01,-6.00549453e-02],
 [-1.92603007e-03,-3.31914984e-04,-2.81903469e-03, 1.35733999e-03,
   1.99588810e-01, 4.99066219e-01, 1.83094655e-02, 2.67997519e-02,
   2.76469958e-02,-5.73440168e-02,-3.79869716e-01]])
    
       
        # Initialize the start observation
        self._old_ob = np.array(10 * np.ones(self.dimension))
        self._current_step = 0
        self._done = False

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def _reset(self):
        self._current_step = 0
        self._old_ob = np.array(10 * np.ones(self.dimension))
        self._done = False

        return self._old_ob.copy()

    def _step(self, action):
        action = np.clip(action, -1., 1.)

        # decide whether to reset the model
        if self._done:
            self._reset()
        feature = np.hstack(([1],self._old_ob,action))
        # get the observation
        ob = np.dot(self.transition, feature) + np.random.multivariate_normal(mean=self.mean, cov=self.identity)

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1

        info = {}
        info['reward'] = reward
        info['current_step'] = self._current_step

        if self._current_step >= self._env_info['max_length']:
            done = True
            self._done = True
        else:
            done = False
            self._done = False

        self._old_ob = ob.copy()
        return ob, reward, done, info

    def reward(self, data_dict):
        # TODO: Change the reward like let the reward be small
        return -0.05 * (np.linalg.norm(data_dict['start_state']) ** 2)


# env_name = 'IVVI'
# environment = env(env_name, 123, {'reset_type': 'gym'})
# tf_env = tf_py_environment.TFPyEnvironment(environment)
# print(isinstance(tf_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())




