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
        self.action_space = box.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.P = 0.5 * np.eye(5) + np.diag([0.2, 0.2, 0.2, 0.2], k=1) + np.diag([0.2, 0.2, 0.2, 0.2], k=-1)
        self.Q = 0.5 * np.eye(5) + np.diag([0.1, 0.1, 0.1, 0.1], k=1) + np.diag([0.1, 0.1, 0.1, 0.1], k=-1)

        self.transition = np.array([[ 5.45257422e-03,  4.75494304e-01,  1.75893276e-01,  2.95158729e-04,   9.52233164e-03,  1.55434405e-02, -4.43286730e-01, -7.03775855e-02,  -5.07090865e-03,  7.59503529e-03, -1.99810524e-02],
 [-4.76899345e-03,  1.81741531e-01,  5.31591539e-01,  2.36797549e-01,   6.19914547e-03, -9.33453809e-03, -6.99442077e-02, -4.38611620e-01,  -1.20564897e-01,  7.61711739e-03,  2.92379631e-02],
 [-5.46138278e-03, -2.32034123e-02,  2.01269498e-01,  5.07356021e-01,   1.85816168e-01, -3.30678072e-03,  6.74552410e-03, -6.75851965e-02,  -4.42100101e-01, -5.90112785e-02,  5.95763595e-03],
 [-2.52159412e-02, -1.96458938e-02,  1.44078223e-02,  1.90380603e-01,   5.09717756e-01,  2.16451251e-01, -7.85473347e-03, -4.74694391e-02,  -8.76692245e-02, -4.41994811e-01, -8.33859561e-02],
 [-1.43580663e-02, -1.61012135e-02,  5.65114764e-02, -1.60841397e-02,   2.28873806e-01,  5.64979995e-01,  1.32880370e-02, -5.26271845e-02,   1.61381509e-02, -1.01488864e-01, -4.58975953e-01]])
        self.mean = np.zeros(5)
        self.identity = np.eye(5)
        # Initialize the start observation
        self._old_ob = np.array([10,10,10,10,10])
        self._current_step = 0
        self._done = False

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def _reset(self):
        self._current_step = 0
        self._old_ob = np.array([10,10,10,10,10])
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




