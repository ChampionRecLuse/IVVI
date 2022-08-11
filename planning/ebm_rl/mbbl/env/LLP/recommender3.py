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
        self.action_space = box.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.transition = np.array([[-1.35432829e-01,  1.01256814e+00,  6.49805208e-03,
         1.72556905e-03,  1.63076240e-03, -2.23675143e-03,
         2.34470059e-03, -4.24883571e-03,  3.07072286e-03,
        -1.26979508e-03,  5.89680893e-03, -9.55541752e-02,
        -7.59324308e-02, -7.29371646e-02, -8.01811744e-02,
         7.38531299e-02,  7.66989611e-02,  8.12070946e-02,
         6.77076642e-02,  7.43655980e-02,  7.03953473e-02],
       [-8.96828815e-02,  4.99819095e-03,  1.00820513e+00,
         2.31034775e-05, -1.83026412e-03, -2.16337437e-03,
         4.27898668e-03, -2.61344491e-03,  1.79498511e-03,
         1.06973000e-03,  2.33844496e-03, -2.99192308e-02,
        -5.14545053e-02, -3.29451691e-02, -4.29853948e-02,
         9.80497492e-02,  1.11115767e-01,  1.06233506e-01,
         9.78301899e-02,  8.91341722e-02,  1.16491525e-01],
       [-1.05333848e-01,  1.56577062e-03,  3.66629166e-03,
         1.00507955e+00,  3.71154277e-03,  2.50542486e-03,
         4.54975689e-03, -5.34607833e-03,  2.28110586e-04,
         1.70528067e-03,  6.05163788e-03, -4.45892306e-02,
        -4.62245822e-02, -6.31447116e-02, -5.60738152e-02,
         9.26718882e-02,  9.85783755e-02,  9.80100601e-02,
         8.81929953e-02,  9.27346604e-02,  8.95277296e-02],
       [-9.89777846e-02,  5.06708444e-03,  8.46794630e-04,
         6.51736733e-05,  1.00334729e+00,  3.69909832e-03,
         4.50344738e-03,  2.24757433e-03, -2.73863234e-05,
         1.97317748e-03,  4.99194759e-03, -4.85985216e-02,
        -4.00585621e-02, -4.07073439e-02, -6.52459362e-02,
         1.01850560e-01,  8.91407197e-02,  1.00478028e-01,
         9.99867905e-02,  9.06390270e-02,  9.99493989e-02],
       [-9.86990949e-02,  9.02939874e-03,  6.25986545e-04,
        -1.66467772e-03, -6.45470750e-04,  1.00986391e+00,
         9.81912403e-03,  5.24835385e-03, -5.80543733e-04,
        -9.54335634e-04, -1.69627188e-03, -3.23891198e-02,
        -3.80344332e-02, -4.20541259e-02, -4.87101212e-02,
         5.33305881e-02,  8.86397465e-02,  9.72127873e-02,
         9.76199376e-02,  8.56085063e-02,  9.51545526e-02],
       [-1.18060041e-01,  7.44080282e-03, -3.14695476e-03,
         1.00272537e-02,  1.57260604e-03, -1.81093105e-03,
         1.00700624e+00,  5.51674754e-03, -1.97963888e-03,
        -5.30385870e-03, -1.42540513e-03, -5.52070977e-02,
        -5.90777037e-02, -5.71150851e-02, -6.26136416e-02,
         7.86701539e-02,  3.50830167e-02,  8.33939479e-02,
         9.19703233e-02,  8.35125271e-02,  8.05061264e-02],
       [-1.01232060e-01,  1.02291245e-02,  3.07425729e-03,
         9.79148952e-03,  1.42274698e-03, -6.63172320e-03,
        -8.90030481e-04,  9.97149616e-01,  1.18021785e-02,
         5.48237378e-03,  1.14001915e-02, -4.46064490e-02,
        -3.97108559e-02, -4.15214703e-02, -5.03641903e-02,
         9.11600965e-02,  9.24231153e-02,  5.64389118e-02,
         9.67461093e-02,  9.51824225e-02,  9.87788922e-02],
       [-1.06106602e-01,  4.84940859e-04,  3.93405476e-03,
         5.11350331e-03,  1.12683547e-03,  2.69682939e-03,
        -4.96763006e-03, -2.79122236e-03,  1.00851480e+00,
         1.90868344e-03, -7.24175257e-04, -4.72028885e-02,
        -4.71292155e-02, -4.59588943e-02, -5.50530396e-02,
         9.08377559e-02,  9.50548263e-02,  9.10854259e-02,
         4.26260612e-02,  8.95132157e-02,  1.01979933e-01],
       [-1.21464045e-01,  1.12912123e-02,  8.73506440e-03,
        -6.89414421e-04, -3.01319998e-04,  1.65557812e-03,
         4.28971061e-03,  2.22040769e-04,  2.69145297e-03,
         1.00417079e+00,  5.79556467e-03, -6.95535846e-02,
        -6.18580457e-02, -5.60847059e-02, -6.42149982e-02,
         7.62725827e-02,  8.60162682e-02,  8.38512339e-02,
         9.24974765e-02,  3.86029172e-02,  8.19153267e-02],
       [-1.29026045e-01,  5.13116272e-03,  7.31493242e-03,
         7.95541389e-03,  9.87411173e-03, -2.18513640e-03,
        -5.22099230e-04,  1.13212304e-03,  9.84241307e-03,
        -4.60546035e-04,  1.00802083e+00, -7.97449324e-02,
        -6.36855601e-02, -7.14925929e-02, -7.41100727e-02,
         7.65362271e-02,  7.92503353e-02,  9.03259271e-02,
         8.34554316e-02,  8.43343203e-02,  2.26052982e-02]])
        
        # Initialize the start observation
        self.U = np.loadtxt('mbbl/env/LLP/preference.txt')
        self.S = np.loadtxt('mbbl/env/LLP/sigma.txt')
        self.movie = np.loadtxt('mbbl/env/LLP/movie.txt')
        self.dimension = 10
        self.Mean = np.zeros(self.dimension)
        self.Identity = np.eye(self.dimension)
        self.number = 2566
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
        action = np.clip(action, 0., 1.)
        action = np.where(action >=0.5, 1, 0)

        # decide whether to reset the model
        if self._done:
            self._reset()
        
        feature = np.hstack(([1], self._old_ob, action))
        # get the observation
        ob = np.dot(self.transition, feature) +  np.random.multivariate_normal(mean = self.Mean, cov= 1 * self.Identity)



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
            utility += rating[i]

        return utility


# env_name = 'IVVI'
# environment = env(env_name, 123, {'reset_type': 'gym'})
# tf_env = tf_py_environment.TFPyEnvironment(environment)
# print(isinstance(tf_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_env.time_step_spec())
# print("Action Specs:", tf_env.action_spec())




