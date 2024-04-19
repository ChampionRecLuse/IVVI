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
         
#         self.transition = np.array([[ 0.20624486,-0.23052063, 0.00626797, 0.01300303,-0.11269471, 0.00479213,
#   -0.00941129, 0.1238145 , 0.01420642, 0.01177756, 0.131282  , 0.13476418,
#   0.13206806],
#  [ 0.20900017, 0.01162765,-0.21957043, 0.012745  , 0.00985927,-0.11464393,
#   -0.00562418, 0.01897077, 0.11049908, 0.01468331, 0.13345401, 0.13181904,
#   0.1354555 ],
#  [ 0.20406649, 0.00313369, 0.01565548,-0.23291882, 0.00117128, 0.01422509,
#   -0.10908428, 0.0152151 , 0.01873479, 0.11846562, 0.12879512, 0.12986753,
#   0.12920479]])
   
        self.transition = np.array([[ 0.21025411,-0.234324  , 0.0070475 , 0.01482084,-0.11935121, 0.00055467,
   0.00574922, 0.12359847, 0.01379496, 0.01551246, 0.12803942, 0.1308524 ,
   0.13536242],
 [ 0.20846922, 0.00697037,-0.230155  , 0.0160707 , 0.00193948,-0.11513947,
  -0.00816898, 0.01935899, 0.11988515, 0.01691628, 0.12669071, 0.13213255,
   0.13173633],
 [ 0.20509758, 0.00703232, 0.01379596,-0.22292047, 0.00607045,-0.00372916,
  -0.11893375, 0.01083498, 0.01374066, 0.12366323, 0.13014391, 0.13046154,
   0.13358691]])
    
        
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




