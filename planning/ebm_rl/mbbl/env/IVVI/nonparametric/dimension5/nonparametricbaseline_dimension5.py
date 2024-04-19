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
         
#         self.transition = np.array([[ 5.57460132e-01,-3.88726630e-01, 1.00528116e-03, 2.97110752e-03,
#   -1.26322684e-03, 6.05288281e-04, 1.19185881e-01,-7.31145313e-03,
#   -3.38245396e-04,-3.36536355e-03, 1.01492177e-02, 1.83646554e-01,
#   8.52812753e-05, 9.98344260e-04, 7.43489185e-05,-3.48653760e-03,
#   4.56214319e-03, 7.32858335e-03,-3.38509857e-03,-2.68825511e-02,
#   1.02659614e-02],
#  [ 5.36645069e-01, 1.48611455e-05,-3.82965570e-01, 7.68921161e-03,
#   3.78856964e-03,-3.24974481e-03,-1.00053045e-03, 1.23458092e-01,
#   -4.59208746e-03,-1.10409903e-02,-5.87286197e-04,-5.64280212e-04,
#   1.77491321e-01,-3.88874285e-03, 1.18115749e-03, 1.18081593e-03,
#   1.66392458e-03,-1.38127674e-02,-1.52462315e-03, 1.87873461e-02,
#   1.38268314e-02],
#  [ 5.37402208e-01, 7.11448434e-03,-5.76968617e-03,-3.80490768e-01,
#   -3.24538493e-03, 7.48299955e-04,-8.00941703e-04, 5.35200304e-03,
#   1.13357109e-01, 1.70217842e-03,-1.09570968e-03,-2.55140410e-03,
#   -1.66388368e-03, 1.79653183e-01, 1.25622143e-03, 1.38689216e-03,
#   1.28946293e-02, 2.49408639e-03,-2.78366221e-03, 1.90481948e-03,
#   -3.34428300e-03],
#  [ 5.29686779e-01,-3.98008941e-03, 4.20100948e-03,-7.37706829e-04,
#   -3.79631391e-01, 1.18413823e-03, 2.10310568e-03, 2.60208059e-03,
#   -8.63940878e-03, 1.20782414e-01,-4.04274982e-03, 1.90729510e-03,
#   -3.00610495e-03, 4.30637271e-03, 1.78054922e-01,-3.77030674e-03,
#   2.22834990e-02,-4.73739646e-03,-4.38261972e-03,-9.18772795e-03,
#   2.01092359e-02],
#  [ 5.37820735e-01, 1.94992597e-03, 8.51916088e-03, 4.26689328e-03,
#   -7.55182893e-03,-3.83752325e-01,-7.86865494e-03, 3.75384224e-03,
#   -7.83766791e-03, 1.02128783e-02, 1.15025910e-01,-2.12219760e-03,
#   5.56197177e-04, 2.24286288e-03, 1.13562627e-03, 1.82417525e-01,
#   1.95527548e-02, 2.95981048e-03,-1.17724056e-02,-1.09882454e-02,
#   7.15541458e-03]])
   
   
        self.transition = np.array([[ 5.39561026e-01,-3.88720997e-01,-1.77966470e-03,-8.85135286e-03,
  -5.90117157e-03, 9.45756923e-03, 1.21995036e-01,-6.21472458e-05,
   1.49437796e-03, 8.24331096e-03,-9.21785286e-03, 1.86881368e-01,
  -4.70210415e-04,-1.48593780e-03, 3.41667743e-03,-4.06430916e-03,
   1.13184703e-02,-7.17006020e-03, 2.06774888e-02,-1.77176007e-03,
  -6.47201746e-03],
 [ 5.65084438e-01, 4.32167999e-03,-3.84242877e-01,-1.92180869e-03,
   3.62104375e-04,-5.54853036e-03, 5.89183768e-03, 1.15863512e-01,
  -8.16455194e-03, 6.08403522e-03, 6.12448131e-03,-1.40487257e-03,
   1.85322659e-01, 2.29659312e-03,-2.07227264e-03, 3.71734907e-03,
  -2.02849778e-02, 3.13296899e-03, 1.52009784e-04,-8.10295884e-03,
  -9.66801389e-03],
 [ 5.29231888e-01,-8.31169014e-03, 2.70693471e-03,-3.81795389e-01,
  -7.71172127e-03, 3.56957671e-03, 2.86627844e-03, 4.87199871e-04,
   1.23051417e-01,-8.64912100e-03, 7.64557994e-03, 3.31272073e-03,
   1.19403261e-03, 1.77808518e-01, 2.55124151e-03, 4.21437880e-03,
  -4.69289629e-03, 5.99932024e-04, 1.06155910e-02, 6.21884735e-03,
   9.38538225e-03],
 [ 5.82082969e-01,-8.63830386e-03, 1.34510306e-03, 4.05413004e-03,
  -3.97705805e-01, 4.00691615e-03,-5.30991194e-04,-1.50884238e-03,
   4.02993800e-03, 1.27825499e-01, 6.88793479e-03, 9.66148038e-03,
  -1.10564946e-03,-3.20287689e-03, 1.81984208e-01,-3.09592958e-03,
   1.10820588e-02,-1.54372762e-02,-2.95636583e-02, 4.04593541e-03,
  -1.39360245e-02],
 [ 5.62457524e-01,-1.21588094e-03, 3.98054912e-03,-2.30501866e-03,
  -3.14807797e-03,-3.87914702e-01, 5.25349217e-04, 9.18247442e-03,
   1.21557412e-04, 4.19922448e-03, 1.26713394e-01, 1.60071818e-03,
  -3.93158414e-03, 9.77778425e-04, 4.60164012e-03, 1.83456629e-01,
  -2.14996725e-02, 7.97849519e-03,-1.28597183e-02, 4.99176686e-03,
  -5.36666590e-03]])
    
        
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




