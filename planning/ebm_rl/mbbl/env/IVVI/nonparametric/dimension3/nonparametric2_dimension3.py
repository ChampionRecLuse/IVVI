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
         
#         self.transition = np.array([[ 0.20229711,-0.23541606, 0.00388111, 0.00771266,-0.18450331, 0.00307825,
#   -0.00502143, 0.11893929, 0.01497798, 0.01159205, 0.13534782, 0.13582554,
#   0.1340383 ],
#  [ 0.20008789, 0.00867148,-0.23033599, 0.00523001,-0.00195472,-0.17633709,
#   -0.00676697, 0.01309811, 0.12729949, 0.00892475, 0.13295552, 0.1346295 ,
#   0.1303362 ],
#  [ 0.20241392, 0.00760078, 0.01163402,-0.2247562 ,-0.00029515, 0.00085072,
#   -0.17245149, 0.01180151, 0.01044114, 0.11888437, 0.13267403, 0.13750339,
#   0.1375873 ]])
   
        self.transition = np.array([[ 2.00164525e-01,-2.29490671e-01, 8.61195335e-03, 1.08061005e-02,
  -1.82013160e-01,-8.69544657e-03,-9.40595997e-03, 1.27756833e-01,
   1.64408505e-02, 1.42806041e-02, 1.27063834e-01, 1.30136155e-01,
   1.32551929e-01],
 [ 2.00199084e-01, 1.78536929e-02,-2.26289637e-01, 8.91835643e-03,
  -4.61205581e-03,-1.74467300e-01, 1.93086337e-04, 1.65587247e-02,
   1.22261587e-01, 1.54882467e-02, 1.33887689e-01, 1.35773767e-01,
   1.23528793e-01],
 [ 2.00568723e-01, 2.27463892e-03, 4.43811532e-03,-2.27698074e-01,
   5.36859458e-03,-5.78171750e-03,-1.74814835e-01, 1.32150704e-02,
   1.33514036e-02, 1.20814185e-01, 1.29459875e-01, 1.38658584e-01,
   1.29132888e-01]])
    
        
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




