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

        self.transition = np.array([[-1.43205235e-01,  9.99961219e-01,  8.68857136e-06,
         3.40736361e-03,  1.21331119e-04,  3.54354513e-03,
         8.39738445e-04,  2.18763232e-03,  4.20985180e-03,
         7.31016251e-03, -1.30469186e-03, -1.01851679e-01,
        -8.29964874e-02, -5.39962745e-03, -3.57271933e-03,
        -2.78567143e-03, -1.87416663e-03,  5.91241883e-02,
         4.88107495e-02,  6.29902214e-02,  4.66978877e-02],
       [-9.45331064e-02,  1.36145763e-02,  1.00649205e+00,
         1.03450209e-02,  3.04757694e-03, -2.21807793e-03,
         2.77906532e-03,  8.18783700e-03, -1.78443539e-03,
         2.76968903e-03,  4.96069591e-03, -3.98858530e-02,
        -5.73390062e-02,  3.12988031e-02,  2.24491470e-02,
         3.88879638e-02,  4.71139110e-02,  9.14117415e-02,
         7.09910947e-02,  9.10075225e-02,  8.59099023e-02],
       [-1.05764336e-01,  1.10997691e-02,  7.54047009e-03,
         1.00328374e+00, -3.69953128e-03, -2.18832609e-03,
         1.09153419e-02, -5.01671979e-04,  2.94467095e-03,
        -4.25952611e-05,  3.05114970e-03, -4.75783025e-02,
        -5.50577144e-02, -1.17585971e-02,  2.45168655e-02,
         2.12309545e-02,  3.53020418e-02,  8.71467860e-02,
         7.22592550e-02,  7.74343673e-02,  6.70983155e-02],
       [-1.02908178e-01, -4.65614532e-06,  7.67593425e-03,
         2.12288111e-03,  1.00719677e+00,  6.21818531e-03,
         6.40372590e-03,  8.08245959e-03,  5.28180070e-03,
        -5.54356868e-03,  9.25315740e-03, -4.58134899e-02,
        -4.45991596e-02,  1.88285254e-02, -1.05798164e-02,
         3.06458142e-02,  2.04661901e-02,  8.84707288e-02,
         7.61263526e-02,  8.91040022e-02,  7.94348813e-02],
       [-1.04607073e-01,  1.16305895e-02,  1.31820943e-03,
         6.66072171e-03, -4.65909597e-03,  1.00908218e+00,
        -6.25134666e-04, -2.62207036e-03,  1.60932550e-03,
        -3.05707216e-03, -3.02295506e-04, -3.98118522e-02,
        -5.17503964e-02,  1.41345844e-02,  1.93451676e-02,
        -9.81797700e-03,  1.21591066e-02,  8.00002913e-02,
         6.76150880e-02,  8.10186428e-02,  7.66069717e-02],
       [-1.29155933e-01,  2.05185155e-04,  4.01239078e-03,
         2.84166231e-03,  4.89557046e-03,  7.21094019e-03,
         1.00923476e+00,  6.12090694e-03,  2.21840221e-03,
        -2.62937550e-03,  2.19688554e-03, -6.40482267e-02,
        -7.20656922e-02,  4.75122247e-03, -3.57997931e-03,
         1.54516196e-02, -2.31034641e-02,  6.74867330e-02,
         6.14734741e-02,  7.12217983e-02,  6.43392965e-02],
       [-1.03056969e-01,  6.89217559e-03,  8.10533093e-03,
         4.44515339e-04,  4.17349434e-03, -2.56435621e-03,
         3.28049943e-03,  1.00089992e+00,  3.39434251e-03,
         4.30973275e-03,  2.97116776e-03, -4.75571677e-02,
        -4.29567488e-02,  2.02372822e-02,  2.14139931e-02,
         2.34581289e-02,  2.27690160e-02,  4.73077847e-02,
         8.20334570e-02,  7.50721657e-02,  7.28435637e-02],
       [-1.04539812e-01,  1.06318728e-02,  1.07900374e-02,
        -1.33357018e-04, -9.33712432e-04,  1.45237060e-03,
         1.12307740e-02,  3.25199588e-03,  1.00945926e+00,
         3.43966713e-03,  1.28485410e-03, -3.85224684e-02,
        -4.91532795e-02,  1.80709907e-02,  1.64257018e-02,
         2.22932079e-02,  2.93122615e-02,  7.27118428e-02,
         2.46516822e-02,  7.85426145e-02,  7.69227960e-02],
       [-1.25849497e-01, -1.06689853e-03, -1.06136194e-03,
        -2.16556383e-03,  5.35607216e-03,  6.81306359e-03,
         5.15739385e-03, -8.41202497e-05,  3.47636725e-03,
         1.00547942e+00,  4.04996876e-03, -6.51031359e-02,
        -6.21910389e-02,  5.43663293e-03,  5.22259695e-03,
         1.33161792e-03,  7.95551215e-03,  5.94092697e-02,
         5.62863488e-02,  2.37443757e-02,  5.28605351e-02],
       [-1.39818305e-01,  4.39596176e-03,  6.30029479e-03,
        -2.07481183e-03,  7.89032560e-03,  8.32111333e-05,
         7.17933675e-03, -4.87862345e-03,  2.82983640e-03,
         4.20331021e-03,  1.00779495e+00, -8.19593927e-02,
        -8.00320889e-02, -6.12238711e-03, -3.82567232e-03,
        -2.08156055e-03, -6.34651043e-03,  7.09114626e-02,
         5.88327925e-02,  6.48361171e-02,  1.32136619e-02]])


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




