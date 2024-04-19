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
         
        self.transition = np.array([[ 1.50356604e-01,-2.25323868e-01, 6.90582683e-03, 6.68327925e-03,
  1.50377484e-02, 3.55499257e-03,-1.00426414e-01, 4.53134707e-03,
  5.64923626e-03, 3.93607410e-03, 8.63107551e-03, 1.16480249e-01,
  1.41655144e-02, 1.20424889e-02, 6.65567617e-03, 1.06156093e-02,
  9.45838807e-02, 9.47991163e-02, 9.74213211e-02, 9.42257380e-02,
  9.47666180e-02],
 [ 1.49194825e-01, 7.43040223e-03,-2.32406964e-01, 4.80429202e-03,
  1.17911988e-02, 9.60234299e-03, 1.19724842e-03,-9.30076291e-02,
  1.40529923e-02, 1.71125071e-03,-4.78832093e-03, 9.15901572e-03,
  1.12409145e-01, 1.14082043e-02, 9.99972428e-03, 3.07558450e-03,
  9.17328313e-02, 9.69920281e-02, 9.91388268e-02, 9.13543693e-02,
  9.82501786e-02],
 [ 1.52550030e-01, 8.22290137e-03, 1.36441552e-02,-2.26366150e-01,
  1.15853920e-02, 1.42810872e-02,-6.38569323e-03, 3.64788499e-03,
  -9.46155640e-02, 7.22279039e-05,-5.18150531e-03, 9.09736055e-03,
  1.47457461e-02, 1.16230250e-01, 3.66111215e-03, 6.77582045e-03,
  9.36706772e-02, 9.55599610e-02, 9.91540278e-02, 9.44740495e-02,
  9.51661867e-02],
 [ 1.53175239e-01, 1.30432048e-02, 8.53363104e-03, 9.91746347e-03,
  -2.26378013e-01, 5.44736389e-03,-8.32932993e-04, 7.53002540e-03,
  4.67009552e-03,-1.01220892e-01, 1.56067780e-03, 8.49988806e-03,
  1.23206294e-02, 7.86686438e-03, 1.17308329e-01, 5.26044034e-03,
  9.49412698e-02, 9.63958738e-02, 9.64737578e-02, 9.28958799e-02,
  9.53231926e-02],
 [ 1.51234940e-01, 2.87721999e-03, 9.41744583e-03, 9.03754161e-03,
  9.23433800e-03,-2.36028521e-01,-1.14367580e-04, 9.39221337e-04,
  2.42402484e-03, 1.67057550e-03,-9.43789836e-02, 9.84214829e-03,
  7.40760767e-03, 8.23005391e-03, 1.04552010e-02, 1.12664489e-01,
  9.26987589e-02, 9.52782867e-02, 9.67281448e-02, 9.46715905e-02,
  9.72888388e-02]])
   
#         self.transition = np.array([[ 0.15206206,-0.23380828, 0.00783527, 0.01016288, 0.01234167, 0.01058056,
#   -0.102677  , 0.00058411,-0.01099701, 0.0002837 ,-0.00264119, 0.11599325,
#   0.00883295, 0.00417742, 0.01351916, 0.01524899, 0.09858692, 0.09637375,
#   0.09807327, 0.09811505, 0.09475847],
#  [ 0.14430716, 0.01041753,-0.23342246, 0.01589308, 0.00270204, 0.00944998,
#   0.00044945,-0.10402585,-0.00293201, 0.00606041, 0.00102367, 0.0114057 ,
#   0.11257691, 0.0042353 , 0.0140318 , 0.00890601, 0.09193686, 0.09251479,
#   0.09350667, 0.09349703, 0.09621892],
#  [ 0.14993317, 0.01060992, 0.01057721,-0.22776394, 0.00966111, 0.00851333,
#   0.00292941,-0.00041843,-0.10282543, 0.00073219,-0.00055265, 0.00989117,
#   0.00610751, 0.1179288 , 0.00817084, 0.00775837, 0.09231163, 0.0948    ,
#   0.09514256, 0.09319636, 0.09584898],
#  [ 0.14848676, 0.00924503, 0.00353991, 0.01084336,-0.2185594 , 0.00736516,
#   -0.00215833,-0.00222684, 0.00180551,-0.0963436 ,-0.00440567, 0.0142491 ,
#   0.00966705, 0.00579589, 0.11371903, 0.01412499, 0.09584532, 0.09363725,
#   0.09513302, 0.09622766, 0.0916717 ],
#  [ 0.14938753, 0.00827735, 0.00712441, 0.00854221, 0.00607093,-0.22489034,
#   0.00596462, 0.00510146,-0.00411654, 0.00118423,-0.10546372, 0.01067159,
#   0.00178823, 0.00729774, 0.01588325, 0.12382555, 0.09129232, 0.09173309,
#   0.09723198, 0.09058087, 0.08933262]])
    
        
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




