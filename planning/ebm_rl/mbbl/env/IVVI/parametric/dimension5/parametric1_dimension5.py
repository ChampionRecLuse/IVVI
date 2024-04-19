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
#         self.transition = np.array([[-4.70989348e-03, 4.97848831e-01, 1.91430661e-01, 4.91614241e-04,
#   -5.84836135e-03,-6.66799375e-03,-3.21914248e-01,-2.78825879e-02,
#   4.22235069e-02, 2.88021680e-02, 3.43567355e-02],
#  [-1.19813314e-03, 2.13829301e-01, 5.04012889e-01, 1.97742290e-01,
#   2.81110337e-03,-6.57875427e-03,-2.49740839e-02,-3.14792512e-01,
#   -4.03360282e-02, 3.92210475e-02, 3.04641209e-02],
#  [ 1.33044992e-02, 1.08965357e-02, 2.05187103e-01, 4.92529796e-01,
#   2.05690141e-01,-3.33509219e-03, 3.85273032e-02,-1.78293606e-02,
#   -3.29232439e-01,-4.11832615e-02, 2.99590134e-02],
#  [-8.87461051e-04, 3.18204511e-03, 1.42125635e-03, 1.97946851e-01,
#   4.76746510e-01, 1.98532003e-01, 3.27085343e-02, 4.37277691e-02,
#   -4.45653065e-02,-3.21137162e-01,-9.96674822e-03],
#  [-2.15743174e-03,-1.61242340e-02, 6.95787410e-04,-1.04558928e-02,
#   1.86495883e-01, 4.93296692e-01, 4.25789268e-02, 3.00204884e-02,
#   3.59122694e-02,-4.22004378e-02,-3.21306569e-01]])
   
        
        self.transition = np.array([[ 0.00345403, 0.49972138, 0.19034444,-0.00303846, 0.0070223 ,-0.01758916,
  -0.31843846,-0.04737885, 0.023449  , 0.03627771, 0.01721919],
 [-0.00420893, 0.20626208, 0.49991502, 0.20135975, 0.01220934,-0.00628562,
  -0.03570584,-0.31865993,-0.03602967, 0.04203792, 0.0345365 ],
 [ 0.00433785,-0.0082471 , 0.18501231, 0.50357276, 0.19731727, 0.00458631,
   0.04855751,-0.03210076,-0.33012864,-0.03040587, 0.03811806],
 [-0.00288952, 0.0008552 ,-0.016151  , 0.20274236, 0.50083535, 0.20037119,
   0.040803  , 0.0369705 ,-0.04267978,-0.31228525,-0.02795565],
 [ 0.00937522,-0.00241955,-0.00178757, 0.0019786 , 0.21351268, 0.49521589,
   0.03483443, 0.03475234, 0.03891713,-0.01905529,-0.32598931]])
    

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




