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
      
        self.transition = np.array([[ 2.60372617e-01,  1.00138618e+00, -5.17084449e-04,
         1.65312905e-03, -2.17348857e-03, -8.56679505e-03,
        -4.50978513e-03, -2.20248339e-03,  4.76321956e-03,
        -2.49718464e-03,  5.60155466e-04, -4.14279466e-01,
        -4.97866148e-02, -8.47245942e-03, -2.34267363e-03,
        -6.18584861e-03,  3.21960016e-03,  8.33366671e-04,
        -2.99618950e-02, -9.89546599e-03, -2.62471547e-02],
       [ 4.10187279e-01, -5.75737206e-03,  1.00330054e+00,
         6.09791718e-04, -3.75393770e-03,  6.38998598e-03,
        -5.76763917e-04,  5.32004026e-03,  7.50849834e-03,
         1.69984771e-03, -6.84657046e-03,  3.92216663e-03,
        -3.50049243e-01,  2.82022379e-02, -3.93353976e-02,
        -1.92234289e-02,  4.48833443e-02, -2.14365580e-02,
        -9.77321130e-03, -3.98732310e-02,  5.20712830e-02],
       [ 3.59625041e-01, -7.49632273e-03, -6.14013327e-04,
         1.00900605e+00, -6.53090914e-03, -4.57488174e-03,
        -5.78308073e-04,  4.15531097e-03, -1.77856067e-03,
         1.09227939e-02, -1.82750457e-03,  4.24100896e-02,
        -4.03140865e-02, -3.92818837e-01,  1.55321730e-02,
         1.33880970e-02,  1.04027332e-02,  3.66429554e-02,
        -2.56246506e-02, -6.38294786e-02, -2.95502207e-03],
       [ 3.14353782e-01, -1.08772772e-03,  4.45594421e-03,
         7.87765556e-03,  9.96413293e-01, -3.25040833e-05,
        -2.59577670e-03,  2.73992598e-03,  9.34068070e-03,
        -5.29601499e-04,  9.75346904e-03, -7.61077798e-04,
         6.02608000e-03,  2.08251168e-02, -3.37560129e-01,
        -1.58490032e-02,  6.70481529e-03,  1.91458980e-03,
         2.63885899e-02,  3.78064960e-02, -2.22269819e-02],
       [ 3.46758775e-01, -2.67182379e-03, -1.73581887e-03,
        -1.13522952e-02, -4.07427984e-03,  9.99538618e-01,
        -1.30974221e-03,  2.26367654e-03, -4.64383980e-03,
         2.71448140e-03, -1.08539681e-02,  9.55709225e-02,
         3.19769997e-02,  7.37123455e-02, -5.38248146e-02,
        -3.38554050e-01, -1.10970539e-01, -3.36979904e-03,
        -1.45227145e-02, -2.93459015e-02, -5.20217946e-02],
       [ 2.87527317e-01, -3.50635166e-03, -3.54148694e-03,
        -1.59643459e-03, -3.98466372e-03, -1.65427078e-03,
         1.00605374e+00, -1.93859633e-03,  1.40600983e-03,
         7.88491760e-04,  4.38155956e-05, -7.14991076e-04,
        -5.09873655e-04, -1.64285957e-02, -4.19801207e-03,
         1.00222101e-02, -3.68909167e-01, -6.48179865e-02,
         2.37848121e-02, -4.81461932e-03, -3.27366899e-02],
       [ 3.40924438e-01,  3.23642122e-03,  2.31290820e-03,
         5.30759841e-04, -3.11490641e-03, -3.69755267e-03,
         5.89985468e-03,  1.00641817e+00,  7.03244521e-03,
         4.61979655e-03, -6.53665896e-03,  2.79741691e-02,
         5.87709335e-02, -1.29680134e-02, -1.41692639e-02,
        -2.39522275e-02, -4.13583435e-02, -3.46408772e-01,
        -2.00425631e-02, -1.03703491e-02, -6.37253904e-03],
       [ 3.65363451e-01,  3.33117688e-03,  6.59987732e-03,
        -9.35796894e-04,  4.81287488e-03,  1.21953680e-03,
         1.06598685e-02, -8.47406035e-03,  9.98360202e-01,
        -4.22209506e-03, -4.33720232e-03,  1.21456952e-02,
        -1.05499108e-02,  5.31777105e-03, -1.95516008e-02,
         7.16673956e-04,  2.69590481e-02, -4.85547137e-02,
        -3.75547927e-01,  1.66068212e-02, -2.05982997e-02],
       [ 3.19181936e-01,  8.87508157e-03,  1.66284001e-03,
        -1.69384038e-03, -5.86373067e-03,  4.06104577e-03,
         5.64581408e-03, -5.74725452e-03, -1.92631459e-04,
         9.99614842e-01,  6.61007616e-04, -4.48237162e-03,
         1.71840998e-02, -2.38649083e-02,  1.30638794e-02,
        -7.40397635e-03, -5.08824475e-02, -2.08099896e-02,
         9.86950026e-03, -3.74205452e-01, -9.65371966e-02],
       [ 3.80976925e-01, -1.07311816e-02,  4.75318889e-03,
         1.86271710e-03,  6.73523247e-03,  7.97223009e-03,
         4.14409355e-03,  5.20095032e-03, -1.18862017e-04,
        -6.21201772e-03,  1.00262186e+00, -6.79127352e-02,
        -1.68976711e-02, -2.60091861e-05,  1.44357004e-02,
        -6.79568013e-02, -6.72899238e-02,  1.89939748e-02,
        -3.53827973e-02, -4.52808411e-02, -4.17195326e-01]])
        
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




