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

        self.transition = np.array([[-1.27283040e-01,  1.01318647e+00,  4.39547831e-03,
         3.41766491e-03, -4.85169720e-05, -1.27872040e-04,
         1.97292423e-04,  5.41706731e-03,  6.19015865e-03,
         1.69420759e-03,  3.09164644e-03, -5.09089933e-02,
         6.21946401e-03,  6.83211803e-03,  7.28606672e-03,
         5.12637656e-03, -3.10951855e-03, -3.61526702e-03,
         4.74692275e-03,  5.75614605e-03,  4.51935205e-04],
       [-8.47614679e-02,  3.51045448e-03,  1.01062336e+00,
         1.31569066e-02,  3.67060052e-04,  7.75083827e-03,
         8.97340985e-03,  1.97423365e-03, -2.02714035e-03,
         1.05400692e-02,  1.04582710e-03,  1.72468939e-02,
         3.15631709e-03,  3.86999710e-02,  3.39859115e-02,
         3.24474575e-02,  2.60651500e-02,  3.29530518e-02,
         2.61201396e-02,  2.58064272e-02,  5.66428612e-02],
       [-1.00290274e-01,  9.77125493e-03,  3.75486852e-03,
         1.00768919e+00,  2.68763335e-03, -2.91628721e-04,
         8.93806119e-03, -3.18067272e-03, -1.11303128e-03,
         6.80349928e-03,  6.38430460e-04,  1.26483203e-02,
         2.31000478e-02, -1.12395797e-02,  2.49248551e-02,
         3.73300360e-02,  2.51398366e-02,  2.34216606e-02,
         3.07555867e-02,  1.97022319e-02,  2.52008561e-02],
       [-9.57250130e-02,  5.00856131e-03,  4.28344467e-03,
         1.46906782e-02,  1.00526211e+00,  1.08281180e-02,
         1.02444352e-02,  1.71410486e-03,  2.40566802e-03,
         5.45215080e-03,  4.59810356e-03,  1.37709542e-02,
         2.50024598e-02,  2.73440323e-02, -5.72629811e-03,
         3.39685503e-02,  2.21398016e-02,  2.52313315e-02,
         3.14324330e-02,  2.72891353e-02,  3.77775311e-02],
       [-9.92035322e-02,  2.58706989e-04,  4.74272995e-03,
         7.18758585e-03, -1.34392824e-03,  1.00607820e+00,
         6.60754903e-03, -2.78994366e-03,  3.71094703e-03,
         1.40985020e-03,  5.42640811e-03,  2.39934707e-02,
         2.27193111e-02,  2.51449336e-02,  2.44482657e-02,
         9.50476694e-04,  8.85298420e-03,  1.48036394e-02,
         2.15047605e-02,  1.80347923e-02,  2.51259736e-02],
       [-1.15100043e-01,  8.61756625e-03,  2.10474734e-03,
        -3.95760082e-04, -1.68277280e-03,  1.12481496e-02,
         1.00570828e+00,  4.83907519e-03,  4.96954335e-03,
        -8.24634548e-03,  2.21759391e-03, -5.77257985e-03,
         1.11489515e-02,  1.40756808e-02,  1.29488610e-02,
         1.78349629e-02, -2.74857196e-02, -2.57281020e-03,
         1.43285572e-02,  1.34641989e-02,  1.36150772e-02],
       [-1.01485668e-01,  8.50519238e-03,  1.50696349e-03,
         1.24568558e-03,  6.53633504e-04,  9.20111194e-03,
         5.26142073e-03,  1.00533758e+00,  1.79900218e-03,
         1.09190284e-02,  1.02286914e-02,  5.22163623e-03,
         2.48525403e-02,  2.99631127e-02,  3.15662564e-02,
         2.43974919e-02,  1.65604421e-02, -1.64124290e-02,
         2.36946382e-02,  2.93627980e-02,  3.47012576e-02],
       [-1.02463634e-01,  5.88852911e-03,  1.55043254e-03,
         8.27361968e-03,  3.82477057e-05,  1.34234012e-03,
         9.20139976e-03, -6.35485458e-04,  1.00990910e+00,
        -1.99797662e-03,  6.72792461e-03,  1.00128451e-02,
         2.34851734e-02,  2.43114117e-02,  3.12192451e-02,
         2.55023162e-02,  1.77145231e-02,  1.42656796e-02,
        -1.16423988e-03,  2.25940319e-02,  2.86800128e-02],
       [-1.23499703e-01, -2.42287058e-03,  2.17833487e-03,
        -5.26598602e-03,  9.89690712e-03,  2.97451440e-03,
        -2.79755488e-03,  2.33075485e-03,  5.39638845e-03,
         1.01003280e+00,  2.19381956e-03, -5.86756531e-03,
         1.38573546e-02,  1.02847488e-02,  1.84358001e-02,
         1.44437895e-02, -7.81517003e-04, -2.52630247e-04,
         1.31001539e-02, -2.53036330e-02,  1.79181427e-03],
       [-1.32199416e-01, -9.18921361e-04,  1.28954719e-03,
         1.11113592e-02, -6.48631128e-03, -1.87561251e-04,
         3.38407796e-03,  3.42892536e-03,  1.16229091e-02,
         5.70376595e-03,  1.01086626e+00, -2.48498310e-02,
         2.27995867e-04,  3.78292827e-03,  8.63370985e-03,
         5.29135049e-03, -8.97971960e-03,  6.82635889e-03,
         4.57569895e-03, -1.81443514e-04, -2.99767225e-02]])
        
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




