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
        self.dimension = 3
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-100, maximum=100, name='observation')
        self.action_space = box.Box(low=-1.0, high=1.0, shape=(self.dimension,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(self.dimension,), dtype=np.float32)
         
#         self.transition = np.array([[ 5.26047410e-01,-3.92384057e-01, 2.90524977e-03, 4.55473857e-03,
#   1.20708833e-01,-9.26423728e-04, 1.12723405e-02, 1.84961624e-01,
#   -8.39106310e-04, 1.34378988e-05, 1.27503236e-02, 1.01487097e-02,
#   -2.75870655e-04],
#  [ 5.72699894e-01, 4.53672798e-03,-3.70955624e-01,-1.76031737e-03,
#   -1.79781373e-03, 1.26232736e-01, 4.54489878e-03, 5.00804853e-04,
#   1.72676012e-01, 7.62123357e-04,-8.99178962e-03,-3.16645431e-02,
#   5.10145863e-03],
#  [ 5.53006705e-01, 3.05711521e-03, 5.05744065e-03,-3.83839306e-01,
#   1.46838071e-04, 6.01472542e-03, 1.16270068e-01,-3.32685470e-03,
#   -2.02377007e-03, 1.79257982e-01,-1.60859625e-02,-1.87735017e-03,
#   2.38830234e-02]])
   
   
        self.transition = np.array([[ 5.70322791e-01,-3.88673814e-01, 7.69469672e-03, 1.28791246e-03,
   1.15243921e-01, 1.26776610e-03,-5.17140130e-03, 1.82422744e-01,
  -4.41476138e-07,-5.16626280e-03, 7.73694145e-03,-1.61278795e-02,
  -1.45195050e-02],
 [ 5.31674882e-01,-1.46389831e-03,-3.72952814e-01,-9.55201703e-03,
   9.10595151e-03, 1.30301081e-01,-1.21967132e-02,-2.81863318e-03,
   1.76729249e-01, 7.50119420e-04, 1.98392432e-02, 1.45638928e-02,
   7.33163198e-04],
 [ 5.29342658e-01,-3.50950221e-05,-6.04946565e-03,-3.67008006e-01,
  -7.01798539e-04, 8.94369440e-03, 1.22214937e-01, 1.46461736e-04,
   3.77872159e-03, 1.72785672e-01, 3.82881114e-03, 1.03822158e-02,
   1.44036895e-02]])
    
        
        # Initialize the start observation
        self._old_ob = np.array(np.zeros(self.dimension))
        self.mean = np.zeros(self.dimension)
        self.identity = np.eye(self.dimension)
        self._current_step = 0
        self._done = False

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def _reset(self):
        self._current_step = 0
        self._old_ob = np.array(np.zeros(self.dimension))
        self._done = False

        return self._old_ob.copy()

    def _step(self, action):
        action = np.clip(action, -1., 1.)

        # decide whether to reset the model
        if self._done:
            self._reset()
        feature = np.hstack(([1], self._old_ob, action, self._old_ob ** 2, action ** 2))
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




