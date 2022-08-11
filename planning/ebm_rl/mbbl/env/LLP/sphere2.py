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
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-100, maximum=100, name='observation')
        self.action_space = box.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.P = 0.5 * np.eye(5) + np.diag([0.2, 0.2, 0.2, 0.2], k=1) + np.diag([0.2, 0.2, 0.2, 0.2], k=-1)
        self.Q = 0.5 * np.eye(5) + np.diag([0.1, 0.1, 0.1, 0.1], k=1) + np.diag([0.1, 0.1, 0.1, 0.1], k=-1)

        self.transition = np.array([[ 0.02363759,  0.51437613,  0.2199663,  -0.02256266, -0.02798426,  0.00278083, -0.38904313, -0.07357111, -0.01076846, -0.00398124,  0.00510107],
 [-0.01416006,  0.19429511,  0.53193271,  0.23036603, -0.02334991, -0.00167302,  -0.04075808, -0.42354252, -0.03898997,  0.02437284,  0.02188576],
 [ 0.03251964,  0.01759076,  0.20837741,  0.53067043,  0.22380858,  0.00537879,   0.01459743, -0.0883077,  -0.37725932, -0.04091348,  0.00629183],
 [ 0.01053009,  0.02462981, -0.00098241,  0.20424284,  0.48521939,  0.22701182,  -0.00168012,  0.00339922, -0.05296529, -0.39200746, -0.07923885],
 [ 0.01253881,  0.04485222,  0.0221239,   0.0078723 ,  0.17959279,  0.49429876,   0.02277778,  0.01785635,  0.01412514, -0.06302663, -0.37448672]])
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




