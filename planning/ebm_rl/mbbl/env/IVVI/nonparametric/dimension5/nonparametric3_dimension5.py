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
         
        self.transition = np.array([[ 1.39637375e-01,-2.39200399e-01, 4.76817382e-03, 9.61188862e-03,
  9.20656055e-03, 3.68994627e-03,-2.15404400e-01,-3.33398091e-03,
  6.30638970e-03,-2.85034118e-03, 1.51790072e-02, 1.27279712e-01,
  6.63880642e-03, 1.14886390e-02, 8.10305530e-03, 4.03564032e-03,
  9.59298790e-02, 9.68733790e-02, 9.50836528e-02, 9.39343459e-02,
  9.15820179e-02],
 [ 1.37168238e-01, 5.91501593e-03,-2.40961735e-01, 1.00742471e-02,
  1.11910388e-02, 3.30508344e-03,-2.77023325e-03,-2.15992732e-01,
  -2.91896927e-05,-7.26841513e-03, 7.64879510e-03, 1.75554024e-03,
  1.22513927e-01, 7.50373985e-03, 8.48673568e-03, 7.99206845e-03,
  8.97388422e-02, 9.23332016e-02, 9.92733748e-02, 9.34964825e-02,
  9.14933817e-02],
 [ 1.34917512e-01, 1.37150428e-02,-4.39874760e-03,-2.36440277e-01,
  3.31018291e-03, 4.63124201e-03,-1.79330392e-03, 1.73599462e-03,
  -2.14742539e-01, 9.35697476e-03,-4.42731426e-03, 3.02249143e-03,
  5.45322066e-03, 1.18640051e-01, 5.33535334e-03, 8.80407323e-03,
  9.35776155e-02, 8.92579401e-02, 9.52335260e-02, 9.49197148e-02,
  8.96879866e-02],
 [ 1.36442615e-01, 4.68952995e-03, 5.45418110e-03, 8.30488330e-03,
  -2.36051802e-01, 3.19781014e-03, 8.08071610e-03, 3.31711038e-04,
  -6.41207639e-03,-2.15827366e-01,-1.08694741e-03, 7.73402038e-03,
  7.43635311e-03, 8.58198497e-03, 1.21101182e-01, 2.45009547e-03,
  9.60828142e-02, 9.24044004e-02, 8.86393944e-02, 9.33667981e-02,
  9.12430086e-02],
 [ 1.38075956e-01, 4.42328558e-03, 9.38717556e-03, 8.59402223e-03,
  -1.97126673e-03,-2.36322168e-01,-2.58187583e-03,-2.53169948e-03,
  -2.00179458e-03, 7.71900512e-03,-2.13918622e-01, 3.83403242e-03,
  1.16747391e-02, 1.32288933e-02, 6.10911946e-03, 1.25755650e-01,
  9.27147375e-02, 9.06935277e-02, 9.42316103e-02, 9.38368404e-02,
  9.22262977e-02]])
   
   
#         self.transition = np.array([[ 1.37088836e-01,-2.41385817e-01, 1.24367803e-03,-4.23366957e-03,
#   -3.26297964e-03, 1.01467009e-02,-2.10843292e-01, 2.06339965e-03,
#   -1.18109041e-03,-2.06906641e-04,-6.33742851e-03, 1.25628349e-01,
#   7.34426746e-03, 3.87487021e-03, 9.79143930e-03, 4.41700698e-03,
#   9.35763423e-02, 9.36490234e-02, 9.21158020e-02, 9.23145728e-02,
#   8.98217959e-02],
#  [ 1.37656892e-01, 4.56452522e-03,-2.44231430e-01, 5.93836744e-03,
#   -2.37937218e-04,-5.82827926e-04,-5.43188761e-04,-2.17003000e-01,
#   -7.29041424e-03, 1.01367867e-02,-2.89073572e-03, 5.84810142e-03,
#   1.31233129e-01, 9.56035557e-03, 1.56674244e-03, 8.88376744e-03,
#   9.55209258e-02, 9.51011080e-02, 9.37001653e-02, 9.22570315e-02,
#   9.28299713e-02],
#  [ 1.35952037e-01, 2.30638863e-03, 5.12549124e-03,-2.32867534e-01,
#   -6.44122545e-03, 5.56857982e-03, 2.71033927e-03,-1.65368849e-03,
#   -2.04367777e-01,-4.26994809e-03,-4.64544602e-03, 1.04578163e-02,
#   8.00739144e-03, 1.16516618e-01, 5.52959374e-03, 1.07988044e-02,
#   9.42410631e-02, 9.47280031e-02, 9.38194184e-02, 9.35649108e-02,
#   9.14197887e-02],
#  [ 1.40159092e-01, 2.96023767e-03, 4.88046356e-03, 7.74863246e-03,
#   -2.46004124e-01, 9.54468391e-03, 9.51095576e-04,-3.12281732e-03,
#   -2.71585079e-03,-2.14521682e-01, 7.30963196e-03, 1.40672970e-02,
#   5.12255863e-03, 6.20995982e-03, 1.23916159e-01, 2.10603238e-03,
#   9.74278940e-02, 9.36206973e-02, 9.48202870e-02, 9.78411600e-02,
#   8.91542175e-02],
#  [ 1.38390327e-01, 3.70369028e-03, 6.60855574e-03, 6.40689914e-03,
#   4.75204949e-03,-2.35298632e-01,-7.02829938e-04, 4.90647758e-03,
#   -3.97967721e-03, 4.49127295e-03,-2.10574924e-01, 9.05163297e-03,
#   3.15891135e-03, 1.05681893e-02, 1.19072897e-02, 1.28470528e-01,
#   9.57876518e-02, 9.72348853e-02, 8.96291682e-02, 9.69438966e-02,
#   9.37777495e-02]])
    
        
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




