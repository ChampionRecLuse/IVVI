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
         
#         self.transition = np.array([[ 1.42832721e-01,-2.34116352e-01, 1.07964846e-02, 1.21799131e-02,
#   3.24151155e-03, 7.37813372e-03,-1.59216777e-01, 2.05411333e-04,
#   6.15789951e-03, 3.80296269e-03,-4.34875696e-03, 1.23682148e-01,
#   7.59600562e-03, 9.07331048e-03, 1.36793678e-02, 8.90379885e-03,
#   9.11273779e-02, 9.23283451e-02, 9.79486166e-02, 9.15543794e-02,
#   9.34703256e-02],
#  [ 1.43018695e-01, 6.27671813e-03,-2.24883764e-01, 3.13584762e-03,
#   4.01497218e-03, 2.18318831e-03,-2.48069804e-03,-1.61332428e-01,
#   2.60932859e-03,-3.26648584e-03,-1.69409638e-03, 8.53208337e-03,
#   1.18202516e-01, 1.23636227e-02, 1.05658964e-02, 1.28194596e-02,
#   9.39239631e-02, 9.26648077e-02, 9.44086844e-02, 9.34475194e-02,
#   9.25712032e-02],
#  [ 1.47325101e-01, 7.02684172e-03, 5.73735314e-03,-2.29292371e-01,
#   9.49657102e-03, 5.42159820e-03,-1.21152119e-03, 5.54837468e-03,
#   -1.66594683e-01,-3.18163152e-03, 7.72466640e-03, 4.64615928e-03,
#   6.49509385e-03, 1.14962698e-01, 7.94949658e-03, 8.21730278e-03,
#   9.07675706e-02, 9.75238091e-02, 9.67582688e-02, 9.68173628e-02,
#   9.91695085e-02],
#  [ 1.42056653e-01, 8.80227021e-03, 9.32968244e-04, 9.70174677e-03,
#   -2.34257066e-01, 1.16596970e-02,-6.18072946e-03, 3.20758045e-03,
#   -2.16536507e-03,-1.58772220e-01,-3.16230485e-04, 1.06436803e-02,
#   7.68003186e-03, 1.06705016e-02, 1.19213283e-01, 8.16249996e-03,
#   9.19947303e-02, 9.57952548e-02, 9.17430455e-02, 9.23948303e-02,
#   9.25597368e-02],
#  [ 1.42312454e-01, 8.36689330e-03, 8.76081854e-04, 1.22052095e-04,
#   9.84437961e-03,-2.31617022e-01, 3.26208924e-03, 1.93483463e-03,
#   5.59932460e-03, 5.94606627e-03,-1.65072869e-01, 1.16085238e-02,
#   1.17830086e-02, 1.02416795e-02, 7.40358925e-03, 1.21099372e-01,
#   9.40869575e-02, 9.59103809e-02, 9.62425743e-02, 9.25645102e-02,
#   9.30099177e-02]])
   
   
        self.transition = np.array([[ 0.14639437,-0.22287179, 0.00649914, 0.009888  , 0.00744432, 0.00458705,
  -0.15691525,-0.00580899, 0.00722245, 0.00368408, 0.00338134, 0.11497726,
   0.00947765, 0.00467099, 0.01004287, 0.00948819, 0.09164092, 0.09606834,
   0.0962382 , 0.09503933, 0.09124502],
 [ 0.14376982, 0.00429019,-0.22889533, 0.00469544, 0.01212065, 0.00109693,
   0.00444678,-0.15952277, 0.00278173, 0.00349886, 0.00312027, 0.01539627,
   0.11649819, 0.00879533, 0.0109363 , 0.00725176, 0.09490916, 0.09551313,
   0.09267561, 0.09380238, 0.09711414],
 [ 0.13942726, 0.00567118, 0.00628931,-0.2350428 , 0.00548261, 0.01378855,
  -0.00370677, 0.00082727,-0.16759601, 0.00103172, 0.01126781, 0.0113855 ,
   0.01250202, 0.11881336, 0.01855324, 0.01077437, 0.09122279, 0.09234598,
   0.09206701, 0.09291143, 0.09304179],
 [ 0.14226324, 0.00909035, 0.00627412, 0.00947086,-0.2334274 , 0.00205705,
   0.00572115,-0.00672854, 0.00183569,-0.16340683, 0.0015119 , 0.00557101,
   0.01190957, 0.00182798, 0.11775092, 0.0100452 , 0.09323161, 0.09567421,
   0.08719522, 0.09449308, 0.09411555],
 [ 0.14777823, 0.00596053, 0.00310579, 0.00868898, 0.01110474,-0.23170286,
   0.00638531, 0.0031556 ,-0.00441523,-0.00680034,-0.16128844, 0.01134623,
   0.00634764, 0.00906201, 0.01221112, 0.12043855, 0.09022637, 0.0995557 ,
   0.09793979, 0.09664487, 0.09546289]])
    
        
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




