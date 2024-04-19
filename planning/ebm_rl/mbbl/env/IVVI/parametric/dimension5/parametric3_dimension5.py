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
#         self.transition = np.array([[-5.49475321e-03, 5.00370111e-01, 2.01969070e-01,-3.31695350e-03,
#   -2.60155896e-03,-5.54777256e-04,-4.17314526e-01,-7.35217186e-02,
#   2.46496166e-02, 2.22938296e-02, 1.52090886e-02],
#  [ 3.15841139e-04, 1.95604852e-01, 5.02126675e-01, 1.96111753e-01,
#   2.48599392e-03,-3.19398647e-03,-6.36672848e-02,-4.27979842e-01,
#   -6.85708869e-02, 1.94057974e-02, 2.26344177e-02],
#  [ 5.16658124e-03,-6.09479924e-03, 2.05286115e-01, 5.08438744e-01,
#   2.08055805e-01, 4.20331778e-03, 9.46793475e-03,-7.42655757e-02,
#   -4.24122113e-01,-6.22596295e-02, 9.56851517e-03],
#  [ 5.13710211e-03,-2.12796473e-03, 1.03417964e-02, 1.98515586e-01,
#   4.96337776e-01, 2.04378660e-01, 1.68738372e-02, 1.43966165e-02,
#   -6.79215619e-02,-4.24872098e-01,-7.99253531e-02],
#  [-7.72163641e-04, 6.36689986e-03, 1.54553271e-02, 4.36587711e-03,
#   1.98406310e-01, 5.08043680e-01, 9.49959018e-03, 1.47177428e-02,
#   1.56940836e-02,-7.76685381e-02,-4.32861841e-01]])
   
   
        self.transition = np.array([[ 0.00491361, 0.50482035, 0.20459898, 0.00644019,-0.00501738,-0.01280878,
  -0.42149906,-0.06654913, 0.01259769, 0.01957198, 0.01443964],
 [-0.00132035, 0.19903243, 0.48521087, 0.2044509 , 0.00732654, 0.00778838,
  -0.0623322 ,-0.41018984,-0.07564194, 0.01078307, 0.010891  ],
 [-0.00436461, 0.00519367, 0.18854782, 0.48923128, 0.20542656,-0.0084361 ,
   0.00904815,-0.06793752,-0.42237037,-0.07268726, 0.01240863],
 [-0.00639833, 0.01612101, 0.00060635, 0.19625632, 0.5065074 , 0.19699512,
   0.02800094, 0.01917069,-0.06845044,-0.42735998,-0.08272009],
 [-0.00214683, 0.00275929, 0.01198852,-0.01993464, 0.18801887, 0.49170681,
   0.02212502, 0.01522033, 0.02429385,-0.0715799 ,-0.4258353 ]])
    

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




