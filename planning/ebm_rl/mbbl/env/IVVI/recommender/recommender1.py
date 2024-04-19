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
from gym.spaces import multi_binary


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

        # self.action_space = multi_binary.MultiBinary(10)
        # self.action_space = box.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = box.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.transition = np.array([[ 2.55308539e-03, 1.01253013e+00,-1.04735500e-02,-5.42403219e-03,
  -1.16547109e-02,-5.78664373e-04,-5.99892483e-03,-7.74883024e-03,
  -1.20793205e-02,-2.20134140e-03, 8.66796982e-03, 2.17710922e-02,
  2.22973016e-02, 1.98453566e-02, 2.49738078e-02, 7.40510605e-02,
  7.11942828e-02, 7.39094914e-02, 7.60529029e-02, 8.07496018e-02,
  6.87531754e-02],
 [ 3.47777530e-02,-3.55869451e-03, 9.90830393e-01,-5.63008097e-03,
  -1.64278933e-02,-6.18164757e-03, 6.24368185e-03,-1.43061721e-02,
  -1.42947485e-02,-5.98225411e-04, 1.70150284e-02, 4.35728567e-02,
  5.21714639e-02, 5.28706313e-02, 5.05390598e-02, 9.21124537e-02,
  9.92369228e-02, 9.50060813e-02, 8.94096140e-02, 9.61259054e-02,
  9.25955327e-02],
 [ 3.48111731e-02, 1.45544859e-02,-1.22498237e-02, 9.89327119e-01,
  -1.92185210e-02,-9.04330528e-03,-1.49673315e-03,-9.89500833e-03,
  -1.62717153e-02,-9.79352409e-03, 1.74098021e-02, 4.80367633e-02,
  5.14053943e-02, 5.45566176e-02, 5.23165708e-02, 8.79465728e-02,
  9.90560922e-02, 9.78230344e-02, 9.02471511e-02, 9.76556242e-02,
  8.87635206e-02],
 [ 3.02383029e-02, 1.18395858e-02,-1.19319396e-02,-5.64300313e-03,
  9.83156022e-01,-2.07687166e-03,-1.97539824e-03, 2.44563861e-04,
  -1.34337045e-02, 2.12220821e-03, 1.16674681e-02, 3.84282398e-02,
  4.73519570e-02, 4.77947092e-02, 5.00649768e-02, 9.16354215e-02,
  9.34723325e-02, 9.43693453e-02, 8.73168560e-02, 9.80690068e-02,
  8.98543619e-02],
 [ 3.18110184e-02, 8.95422333e-03,-1.83090376e-02,-2.57803434e-03,
  -2.54438731e-02, 9.83008808e-01,-8.05351118e-03,-4.17155794e-03,
  -1.52955917e-02, 5.48074868e-03, 6.92962476e-03, 4.76554301e-02,
  4.76952356e-02, 4.74261333e-02, 4.94924262e-02, 9.68715333e-02,
  8.93297722e-02, 9.16175882e-02, 8.95893957e-02, 9.13910354e-02,
  8.86772882e-02],
 [ 1.17863551e-02, 1.40342198e-02,-1.32576079e-02,-4.74080462e-03,
  -9.82128005e-03,-2.92487453e-03, 9.90505068e-01,-5.91497106e-03,
  -1.38528276e-02, 2.74816603e-03, 1.17375486e-02, 2.80594969e-02,
  2.82099106e-02, 2.73954802e-02, 3.03269261e-02, 7.52921022e-02,
  8.91561957e-02, 7.87555977e-02, 7.94037644e-02, 7.75364428e-02,
  8.53456204e-02],
 [ 1.55489702e-02, 5.29023507e-03,-8.09825530e-03,-1.58018394e-02,
  -2.78243966e-02,-2.20021751e-02, 3.86282172e-03, 9.99314334e-01,
  -1.44019375e-02,-5.69884290e-04, 1.73393707e-02, 3.05554411e-02,
  3.06769898e-02, 2.90370616e-02, 3.45174304e-02, 8.57563108e-02,
  8.34444870e-02, 9.84063772e-02, 7.72342734e-02, 8.58815643e-02,
  7.67366562e-02],
 [ 3.43700126e-02, 1.49494261e-02,-1.15903534e-02,-9.06323622e-03,
  -1.74776162e-02,-8.61854020e-03,-9.72345895e-03,-2.03381103e-02,
  9.68441818e-01, 4.40164188e-03, 1.44007843e-02, 5.25290600e-02,
  5.35039815e-02, 4.85793761e-02, 5.04769746e-02, 9.18204904e-02,
  9.54214776e-02, 9.48564560e-02, 1.01946831e-01, 1.01592002e-01,
  9.36411404e-02],
 [-7.19530893e-03,-1.26729011e-03,-7.98333879e-03, 2.37312764e-03,
  -1.67296411e-02,-1.49761162e-02,-7.37735984e-03,-3.66952641e-03,
  -1.65730060e-02, 1.00018832e+00, 1.15049709e-02, 9.39179030e-03,
  1.22134378e-02, 1.28362553e-02, 1.22128372e-02, 6.24299122e-02,
  7.34081874e-02, 7.04569607e-02, 5.62558096e-02, 7.98914372e-02,
  7.08573927e-02],
 [-1.38030831e-02, 8.96638869e-05,-7.31067062e-03,-9.51479861e-03,
  -1.11928728e-02,-1.01764880e-02, 3.74646170e-03,-5.94139093e-03,
  -1.06448861e-02, 4.22684774e-03, 1.01738823e+00,-1.56645745e-03,
  7.41983076e-03, 6.60001055e-03, 8.86468732e-03, 6.30251635e-02,
  6.62707807e-02, 7.00476243e-02, 5.89189056e-02, 7.38662527e-02,
  7.00806770e-02]])
   


        
        # Initialize the start observation
        # need to change the path when training
        self.U = np.loadtxt('/content/gdrive/MyDrive/estimation_revise/preference.txt')
        self.S = np.loadtxt('/content/gdrive/MyDrive/estimation_revise/sigma.txt')
        self.movie = np.loadtxt('/content/gdrive/MyDrive/estimation_revise/movie.txt')
        self.dimension = 10
        self.Mean = np.zeros(self.dimension)
        self.Identity = np.eye(self.dimension)
        self.number = 2566
        # self._old_ob = self.U[np.random.randint(0,6040),:]
        # self._old_ob = self.U[self.number,:]
        self._old_ob = np.zeros(10)
        self._current_step = 0
        self._done = False

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def _reset(self):
        self._current_step = 0
        # self._old_ob = self.U[self.number,:]
        self._old_ob = np.zeros(10)
        self._done = False

        return self._old_ob.copy()

    def _step(self, action):
        # action = np.clip(action, -1., 1.)
        # action = np.where(action >=0, 1, 0)
        action = np.clip(action, 0., 1.)
        action = np.where(action >=0.5, 1, 0)

        # decide whether to reset the model
        if self._done:
            self._reset()
        
        feature = np.hstack(([1], self._old_ob, action))
        # get the observation
        ob = np.dot(self.transition, feature) +  np.random.multivariate_normal(mean = self.Mean, cov= 1 * self.Identity)
        # ob = np.dot(self.transition, feature) +  0.1 * np.random.multivariate_normal(mean = self.Mean, cov= 1 * self.Identity)



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
        rating = np.clip(np.dot(self.S,np.dot(self.movie,data_dict['start_state'])),0,5)
        utility = 0
        for i in range(self.dimension):
            # if data_dict['action'][i] == 1:
                # utility += rating[i] - 2.5
                # utility += rating[i]
            utility += rating[i]

        return utility


# env_name = 'IVVI'
# environment = env(env_name, 123, {'reset_type': 'gym'})
# tf_env = tf_py_environment.TFPyEnvironment(environment)
# print(isinstance(tf_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())




