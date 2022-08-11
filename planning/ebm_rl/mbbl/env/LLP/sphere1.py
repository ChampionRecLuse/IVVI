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

        self.transition = np.array([[-1.99988263e-02,  4.92696493e-01,  1.99671481e-01 ,-3.42009004e-02 , -1.36551374e-02 ,-2.20559222e-02, -2.97905867e-01, -4.01797939e-02,   3.62196420e-02 , 3.10036798e-02,  4.38392011e-02],
 [ 8.45456975e-03,  1.83159096e-01,  5.23964611e-01,  2.16948938e-01,  -8.35738634e-03, -2.48592368e-03, -5.61845536e-02, -3.27696478e-01,  -5.28383695e-02,  3.30596517e-02,  1.23183501e-02],
 [ 3.59684590e-03, -1.83484303e-02,  2.13120823e-01,  5.27320630e-01,   1.92447743e-01, -1.86307447e-02,  3.15195980e-02, -6.98310327e-02,  -3.15578006e-01, -3.19887174e-02,  3.23131868e-02],
 [ 2.34493661e-05, -3.35121913e-02,  3.26731735e-02,  1.63709928e-01,   4.90175965e-01,  1.68368261e-01,  3.16743008e-02,  2.17818583e-02,  -3.10173059e-02, -3.17332624e-01, -4.43586273e-02],
 [-3.09305794e-02,  1.51642281e-02,  1.76537453e-02, -1.98458334e-02,   2.31687424e-01,  5.01276693e-01,  3.19736558e-02,  1.27763121e-02,   2.22025764e-03, -4.72725168e-02, -3.27885256e-01]])
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




