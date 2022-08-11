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

        self.transition = np.array([[ 4.61858930e-03,4.91521067e-01,1.96333599e-01,1.62279275e-03,6.03502655e-03,6.25824973e-03,1.92586698e-01,-2.69441841e-02,-1.54009826e-02,9.03028980e-03,1.33136488e-02],
 [-5.85420005e-03,1.99354920e-01,4.95725890e-01,1.97772313e-01,7.63434414e-03,-1.82963934e-03,-1.61103371e-02,1.64484893e-01,-2.41020670e-02,-2.52232241e-02,7.97209415e-03],
 [-1.17777020e-03,-7.06594256e-03,2.05151564e-01,5.04587358e-01,1.98089021e-01,-5.37167062e-03,-1.55879349e-02,-2.97980183e-02,1.68102076e-01,-1.85741600e-02,-7.52482280e-03],
 [-4.63853266e-04,4.12265083e-03,-1.28419300e-02,2.09926533e-01,4.94598785e-01,1.97892017e-01,-8.24493980e-04,-2.50967205e-02,-2.30290956e-02,1.84142503e-01,-1.83025138e-02],
 [-1.38431625e-02,7.78646611e-03,-1.42067498e-02,7.31398017e-03,1.96643859e-01,5.08247353e-01,3.87824776e-03,-1.93947792e-02,-5.85790106e-03,-1.21297178e-02,1.81274112e-01]])
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




