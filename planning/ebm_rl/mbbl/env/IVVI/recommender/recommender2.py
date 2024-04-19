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
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-100, maximum=100, name='observation')
        # self.action_space = array_spec.BoundedArraySpec(shape=(10,), dtype=np.int32, minimum=0, maximum=1, name='action')
        # self.action_space = box.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        # self.action_space = multi_binary.MultiBinary(10)
        # self.action_space = box.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = box.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.transition = np.array([[-1.51957268e-02, 1.00665305e+00,-2.37030976e-02,-8.17908304e-03,
  1.09072073e-03,-3.49874859e-03,-5.59262686e-03,-4.08878222e-03,
  1.31931264e-02, 2.68163122e-03, 2.28892794e-02, 5.43816397e-03,
  7.34434349e-03, 3.70907578e-02, 3.49783517e-02, 3.50297341e-02,
  2.85669308e-02, 5.51198169e-02, 5.52899118e-02, 5.40647253e-02,
  5.99319831e-02],
 [ 4.91167954e-02, 1.28399016e-02, 9.73205984e-01,-2.06844778e-02,
  -6.20358761e-04, 5.16715871e-03, 2.42746059e-03,-2.21360907e-03,
  1.19250288e-02,-1.34923855e-02, 1.85838066e-02, 6.04586291e-02,
  6.90975136e-02, 8.37239681e-02, 9.17465090e-02, 7.98039568e-02,
  8.87860230e-02, 9.48180693e-02, 9.86945500e-02, 9.12181292e-02,
  1.03020881e-01],
 [ 2.84895255e-02, 2.10044770e-02,-2.11415248e-02, 9.91247406e-01,
  -3.88605842e-03,-1.38982436e-02,-2.91477232e-03,-7.69845138e-03,
  1.49040014e-02,-7.44675211e-03, 1.71670797e-02, 4.05700584e-02,
  4.14433252e-02, 6.82544632e-02, 7.03434766e-02, 6.41791328e-02,
  6.83827088e-02, 7.90478624e-02, 8.65951697e-02, 7.89377211e-02,
  8.60779190e-02],
 [ 3.01978053e-02, 8.84352595e-03,-1.62666213e-02,-1.03335210e-02,
  9.88786834e-01, 7.48360096e-03,-5.04798896e-03,-7.71699047e-03,
  2.28223618e-03,-1.88862252e-03, 9.65546623e-03, 4.49095033e-02,
  4.69376710e-02, 7.32936858e-02, 7.54776403e-02, 6.51823745e-02,
  6.33663587e-02, 8.09560961e-02, 8.48062878e-02, 8.03766604e-02,
  7.93179977e-02],
 [ 3.72003832e-02, 7.71760606e-03,-1.77428211e-02, 6.15753378e-03,
  -7.21047841e-03, 9.89965967e-01,-7.00218279e-03,-9.38598374e-03,
  2.94692056e-04, 2.84498837e-04, 1.76842387e-02, 5.00469639e-02,
  4.98192555e-02, 7.61731863e-02, 7.52776836e-02, 7.66510302e-02,
  7.02453082e-02, 8.76869319e-02, 8.93028277e-02, 8.43054291e-02,
  8.93376766e-02],
 [ 3.66157311e-03, 1.56749848e-02,-2.75225462e-02,-9.96997335e-03,
  2.17169853e-03,-1.57737699e-04, 9.84301637e-01,-6.72721306e-03,
  1.67659899e-02,-2.93280386e-03, 2.15740207e-02, 2.25693671e-02,
  2.39928748e-02, 5.45934164e-02, 4.64305839e-02, 4.78300091e-02,
  4.98352557e-02, 6.19516382e-02, 6.29826078e-02, 6.84579690e-02,
  6.54654860e-02],
 [ 1.74659849e-02, 8.84597177e-03,-1.61885202e-02,-9.39901160e-03,
  -9.82255175e-03,-7.23959196e-03, 2.80232633e-03, 9.98026755e-01,
  4.65083886e-03,-1.49153303e-02, 3.89896953e-03, 2.75553651e-02,
  3.34968577e-02, 5.76455230e-02, 5.95278653e-02, 5.71410851e-02,
  5.92596050e-02, 8.11120068e-02, 7.33373271e-02, 7.99254803e-02,
  7.66858010e-02],
 [ 2.58896456e-02, 1.58320280e-02,-1.73278541e-02,-8.36042444e-03,
  5.13422906e-03,-7.89439217e-04,-8.73894477e-03,-1.14918741e-02,
  1.00896221e+00,-1.66469502e-02, 1.94059457e-02, 3.67406216e-02,
  4.21795875e-02, 6.75106774e-02, 6.73224708e-02, 6.22768057e-02,
  6.50465693e-02, 8.02863895e-02, 8.60479954e-02, 8.19393807e-02,
  8.22481801e-02],
 [-1.07091400e-03, 8.22187383e-03,-1.18539280e-02,-3.67598425e-03,
  -6.80107227e-03,-7.65709491e-03, 2.07182459e-03,-5.55765128e-03,
  -8.70187136e-04, 9.98871776e-01, 1.17694881e-02, 1.42066862e-02,
  1.73665353e-02, 4.32167861e-02, 4.60205876e-02, 4.03439194e-02,
  4.39776942e-02, 6.27555409e-02, 6.63488899e-02, 6.63882783e-02,
  6.49853029e-02],
 [-3.04841181e-03, 1.16035256e-02,-2.10058606e-02,-8.02297479e-03,
  -1.06383245e-03, 6.55104763e-03, 7.15443715e-04,-8.94854132e-04,
  2.56551368e-03,-2.44106624e-03, 1.01793815e+00, 1.34550744e-02,
  1.56108024e-02, 4.39570957e-02, 4.32467858e-02, 4.41090258e-02,
  3.82502522e-02, 6.13276191e-02, 6.40287811e-02, 6.82166249e-02,
  7.61357337e-02]])
   
   


        # Initialize the start observation
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




