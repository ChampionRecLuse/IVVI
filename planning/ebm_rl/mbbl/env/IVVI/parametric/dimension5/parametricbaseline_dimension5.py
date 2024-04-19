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
#         self.transition = np.array([[-7.84797226e-04, 5.02772691e-01, 1.95935527e-01, 4.83361701e-03,
#   -3.89384218e-03, 3.69171427e-03, 4.37994938e-02,-7.80065798e-03,
#   -8.60147365e-03, 6.20935777e-03, 1.77959447e-03],
#  [ 4.78417894e-03, 1.99307212e-01, 4.99575547e-01, 1.98105011e-01,
#   3.42443482e-03,-1.12918881e-03,-9.04050362e-03, 3.02961435e-02,
#   -4.65465317e-03,-1.52298764e-02, 4.49771648e-03],
#  [ 1.34724125e-03, 3.96806419e-03, 2.00648601e-01, 4.98874455e-01,
#   1.98123707e-01,-1.96294239e-04,-1.51318027e-02,-3.86537805e-03,
#   3.59150227e-02,-2.19014822e-03,-1.72280015e-02],
#  [ 2.37264398e-03,-1.62691915e-04, 3.32331955e-03, 1.97228168e-01,
#   4.97649226e-01, 2.02240258e-01,-2.51084079e-03,-6.79151310e-03,
#   -5.91922003e-03, 3.53672738e-02,-1.04266599e-02],
#  [ 2.06207794e-03,-1.61180517e-03, 7.60070460e-03,-5.36706192e-03,
#   1.99140549e-01, 4.98564074e-01,-4.10028610e-03, 1.33812095e-03,
#   -1.24272283e-02,-1.27679867e-02, 4.62541450e-02]])

        self.transition = np.array([[-1.52277286e-03, 4.99058021e-01, 2.04830648e-01, 1.72245492e-03,
   2.00402044e-03,-2.57056332e-03, 4.89146279e-02,-1.21347533e-03,
  -1.06846914e-02, 2.48910073e-03, 1.17525716e-03],
 [ 3.49858802e-03, 1.98522669e-01, 4.99450282e-01, 1.99129449e-01,
   6.61912340e-03,-1.97749428e-03,-4.57930293e-03, 4.01432702e-02,
  -1.60339675e-03,-1.50726169e-02,-3.42217210e-03],
 [ 1.32923958e-03, 1.39506091e-03, 1.98640812e-01, 4.97350288e-01,
   2.06649474e-01,-3.62619517e-03,-1.12377124e-02, 9.10306162e-03,
   3.65840930e-02,-6.26420037e-03,-1.41110749e-02],
 [-3.02201834e-04, 3.44140641e-03, 1.06260236e-03, 1.94350031e-01,
   5.08999625e-01, 1.96977139e-01, 9.06553944e-03,-8.93066454e-03,
   2.71871462e-03, 3.78386187e-02,-6.31127593e-03],
 [ 1.83679073e-03,-3.74384321e-04, 5.24690082e-03,-3.47890159e-03,
   1.98843075e-01, 4.97812318e-01, 3.81590305e-03, 5.46721695e-04,
  -8.43446955e-03,-4.17260184e-03, 4.94872484e-02]])
    
        
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




