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
        self.dimension = 10
        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.int32, minimum=-100, maximum=100, name='observation')
        self.action_space = box.Box(low=-1.0, high=1.0, shape=(self.dimension,), dtype=np.float32)
        self.observation_space = box.Box(low=-np.inf, high=np.inf, shape=(self.dimension,), dtype=np.float32)
        self.P = 0.5 * np.eye(self.dimension) + np.diag(0.2 * np.ones(self.dimension-1),k=1) + np.diag(0.2 * np.ones(self.dimension-1),k=-1)
        self.Q = 0.5 * np.eye(self.dimension) + np.diag(0.1 * np.ones(self.dimension-1),k=1) + np.diag(0.1 * np.ones(self.dimension-1),k=-1)
        self.mean = np.zeros(self.dimension)
        self.identity = np.eye(self.dimension)
#         self.transition = np.array([[ 4.18833623e-03, 5.03252208e-01, 2.09749012e-01, 1.01953709e-02,
#   -1.73830386e-02,-1.37839125e-02,-3.52132429e-03, 4.49948534e-03,
#   1.61970411e-03, 6.37793038e-03, 7.02880982e-03,-4.09900583e-01,
#   -7.28723522e-02, 1.35195279e-02, 1.30695186e-02, 2.64704882e-02,
#   4.43740125e-03, 1.54456422e-02, 1.04139104e-02, 1.07551376e-02,
#   1.21898420e-02],
#  [ 5.70990340e-03, 2.00217372e-01, 5.17095588e-01, 2.04351960e-01,
#   5.08047681e-03,-5.99432476e-04,-7.92732563e-03, 1.10899409e-02,
#   -5.18904147e-03, 3.45162473e-03,-6.85765508e-04,-7.32667411e-02,
#   -4.15932745e-01,-7.23081170e-02, 4.37132761e-03, 1.83126854e-02,
#   8.73782574e-03, 1.71995288e-02, 1.99592040e-02, 1.06509958e-02,
#   1.05721870e-02],
#  [ 3.24711257e-03, 4.06371871e-03, 2.06330634e-01, 5.03426917e-01,
#   2.12007835e-01, 8.10903445e-03,-1.44296245e-02, 9.93178217e-03,
#   -7.80293733e-03, 3.25377012e-03, 4.15370795e-03,-2.31267571e-04,
#   -6.76754019e-02,-4.03593244e-01,-7.14957442e-02, 9.32595606e-03,
#   4.57852319e-03, 1.95323219e-02, 1.07473116e-02, 1.50953254e-02,
#   7.64716442e-03],
#  [ 9.82310866e-04,-6.84344046e-03, 1.51974333e-02, 1.98260558e-01,
#   5.16108835e-01, 2.20842004e-01, 1.85413937e-04, 4.86135002e-03,
#   3.46222155e-04, 1.64701572e-02,-1.09869947e-03, 1.33366638e-02,
#   1.51714064e-02,-6.29043507e-02,-3.95345610e-01,-8.35686632e-02,
#   1.13245909e-02, 1.07208656e-02,-3.39570341e-03, 5.43735747e-03,
#   9.56846587e-03],
#  [ 4.75087754e-03, 5.85582093e-03,-6.36017630e-03,-9.39006054e-04,
#   1.99131197e-01, 5.13461855e-01, 2.01021079e-01, 1.01548284e-02,
#   8.91145210e-03, 1.89661271e-02, 4.41314168e-03, 1.08657307e-02,
#   1.34890093e-02, 1.40920625e-02,-7.03195889e-02,-4.09576120e-01,
#   -6.24883883e-02, 2.53509734e-02, 1.07505884e-02, 2.77780587e-03,
#   1.40083619e-02],
#  [ 4.50842681e-03, 7.25680785e-03,-2.69586089e-02,-1.27023115e-02,
#   -1.30368101e-02, 2.05333155e-01, 5.02538397e-01, 2.03660818e-01,
#   -5.08064775e-04, 1.09352220e-02, 6.60597040e-03, 1.51933552e-02,
#   1.99068728e-02, 1.07621755e-02, 1.64343289e-02,-5.74895743e-02,
#   -3.97912625e-01,-6.36642971e-02, 1.19422925e-02, 1.29146566e-02,
#   2.04702050e-02],
#  [-6.42197803e-03, 4.25888055e-03,-1.51803860e-02,-5.29111533e-03,
#   4.95501149e-03,-2.06849556e-03, 2.12558999e-01, 4.93865605e-01,
#   1.95834372e-01, 3.23304293e-03, 7.10993417e-03, 9.73138984e-03,
#   1.65696033e-02, 1.39444473e-02, 7.90501091e-03, 1.93299768e-02,
#   -7.12909398e-02,-3.94661591e-01,-7.05372127e-02, 6.26598782e-03,
#   3.89507480e-03],
#  [-1.24701825e-02,-3.34248199e-03, 4.11418155e-03,-2.83737552e-04,
#   1.29509160e-02,-1.35023310e-03,-2.90651627e-06, 1.97298146e-01,
#   4.97245363e-01, 1.87262851e-01, 4.51294919e-04,-8.28024357e-03,
#   3.77016535e-03, 8.72893646e-03, 1.21249353e-02, 4.11151390e-03,
#   2.15254559e-02,-7.30209996e-02,-3.95850189e-01,-7.61737045e-02,
#   -1.49344456e-03],
#  [-5.51366489e-03,-2.22773012e-02,-1.31166887e-02,-1.07446833e-02,
#   1.19137013e-02,-9.28920997e-03,-2.14292054e-03, 5.79157011e-04,
#   1.93128350e-01, 4.89105939e-01, 2.06662396e-01, 2.48182927e-02,
#   -1.09219226e-03, 9.87769574e-03, 1.24132094e-02, 1.05632336e-02,
#   2.02216034e-02, 1.38674103e-03,-6.30401085e-02,-4.05597072e-01,
#   -7.03904330e-02],
#  [ 5.71200109e-03, 8.32145717e-03, 7.93836131e-03,-9.02919684e-03,
#   6.00029258e-03,-9.39064510e-03,-2.88057687e-03, 1.11877526e-02,
#   4.02399091e-04, 1.96684637e-01, 5.08397702e-01, 8.87330745e-03,
#   1.98556585e-02, 1.61724998e-02, 2.17396707e-02, 1.77510645e-02,
#   6.19896757e-03, 6.73093086e-03, 2.45403136e-02,-5.52820189e-02,
#   -4.06282410e-01]])

        
        self.transition = np.array([[ 3.09837067e-03, 5.04450439e-01, 2.06204874e-01,-2.95146964e-03,
  -5.54299537e-03,-8.65145164e-03,-2.09832535e-03,-6.07788423e-03,
   1.07083527e-02,-3.34465312e-03,-1.29613244e-02,-3.98833100e-01,
  -6.08839336e-02, 1.93137363e-03, 7.52485444e-03, 1.03330011e-02,
   1.49980983e-02, 7.58215589e-03, 9.87120974e-03, 1.09222820e-02,
   7.57431847e-03],
 [ 7.65559931e-05, 1.99341998e-01, 5.05156796e-01, 2.00748531e-01,
   2.85421142e-04, 4.62304723e-03, 1.00156474e-02, 3.35043210e-03,
  -1.10751842e-02, 5.62214267e-03,-7.99139061e-04,-5.75420714e-02,
  -3.95860912e-01,-6.48158507e-02, 1.74579931e-02, 2.07443877e-03,
   5.61030432e-03, 6.22795623e-03, 4.10444408e-04, 1.11311048e-02,
  -3.41168006e-04],
 [ 1.92742374e-03, 1.73903937e-04, 1.96165829e-01, 5.09379908e-01,
   2.02055047e-01,-2.46728593e-03,-1.82801587e-02, 5.33853818e-03,
  -5.06163361e-04, 3.75728162e-03, 8.13224974e-03, 1.78931593e-02,
  -7.06780252e-02,-4.19869645e-01,-4.96624611e-02, 1.01523390e-02,
   6.49579825e-03, 9.89369365e-03, 1.29918900e-02, 1.96796681e-02,
   1.15523433e-02],
 [-8.43150077e-03,-5.79522149e-03,-2.99408408e-03, 2.11767551e-01,
   4.93141390e-01, 1.89698633e-01,-1.21328026e-02, 9.44879816e-03,
  -7.66031609e-03, 3.24389863e-03, 5.03061375e-03, 1.94404865e-02,
   1.41517787e-02,-7.18208915e-02,-4.05376560e-01,-5.82801564e-02,
   2.15199879e-02, 9.13564903e-03, 3.85477863e-03, 2.55880193e-02,
   1.49976807e-02],
 [-2.06497882e-03,-2.56004734e-03,-9.77970298e-03,-1.35106726e-02,
   1.86336213e-01, 4.89460952e-01, 1.98949634e-01, 1.05028611e-02,
   9.56782613e-03, 1.05740544e-02, 6.67367801e-03, 1.01822492e-02,
   1.67995723e-02, 1.02744029e-02,-7.04980502e-02,-3.97448853e-01,
  -5.57905110e-02, 1.78641808e-02, 1.36175388e-02, 3.25860046e-03,
   2.80118224e-02],
 [-8.07831051e-03, 4.45212105e-03,-1.15185761e-03, 8.76652392e-03,
  -1.44482470e-03, 1.96326452e-01, 4.96332890e-01, 2.11914649e-01,
   4.12665690e-03, 5.65274399e-03,-1.66398716e-02, 7.87446198e-03,
   1.58782872e-02, 1.86227194e-02, 9.24177048e-03,-6.99559311e-02,
  -3.98317840e-01,-6.75758019e-02, 1.44007641e-02,-4.15486126e-03,
   1.30738195e-02],
 [-6.45624005e-03, 6.08882784e-03,-4.19282865e-03,-9.19450457e-04,
   1.04524713e-03,-1.19240500e-02, 1.98727459e-01, 5.05208237e-01,
   1.86573617e-01, 2.02772868e-03,-1.02476568e-02, 8.89250731e-03,
   8.57094523e-03, 1.63955592e-02, 1.61189608e-02, 1.04833277e-02,
  -6.75026723e-02,-4.02785900e-01,-5.68647074e-02, 4.61067567e-03,
   1.26486965e-02],
 [-1.04138125e-02,-1.21870957e-02, 8.58441545e-04, 1.77713898e-03,
  -5.15491912e-03, 2.07848409e-04, 8.23657677e-03, 2.03824155e-01,
   4.88462087e-01, 2.00547056e-01,-5.11561301e-03, 1.75746025e-02,
   5.90635059e-03, 2.40965277e-03, 2.02716632e-02, 1.01988117e-02,
   1.15404466e-02,-6.51255104e-02,-4.10544283e-01,-7.41793848e-02,
   2.04705912e-02],
 [ 4.12372081e-03,-3.43393398e-03, 1.31095538e-02, 1.01110524e-02,
   4.93285658e-03, 8.96133958e-03,-5.43849075e-04,-6.00096666e-03,
   2.00592437e-01, 4.94931873e-01, 2.02914075e-01, 1.24959837e-02,
   1.49421159e-02, 1.71783374e-02, 1.92341766e-02, 9.44268115e-03,
   2.28387098e-02, 1.06641052e-02,-8.00953283e-02,-4.03514468e-01,
  -7.26205938e-02],
 [ 7.84792489e-03, 2.53145417e-03, 2.59735995e-03, 2.27628065e-02,
   7.62403947e-03, 4.58891174e-03,-8.92352596e-03, 3.70889714e-03,
   1.58317467e-03, 1.97987359e-01, 5.11028288e-01, 4.03783578e-03,
   1.24716620e-02, 1.66018976e-02, 7.19127787e-03, 1.18021949e-02,
   1.93900174e-02, 8.36492610e-03, 1.04494060e-02,-7.63004664e-02,
  -4.02353799e-01]])
    

        # Initialize the start observation
        self._old_ob = np.array(10 * np.ones(self.dimension))
        self._current_step = 0
        self._done = False

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def _reset(self):
        self._current_step = 0
        self._old_ob = np.array(10 * np.ones(self.dimension))
        self._done = False

        return self._old_ob.copy()

    def _step(self, action):
        action = np.clip(action, -1., 1.)

        # decide whether to reset the model
        if self._done:
            self._reset()
        feature = np.hstack(([1],self._old_ob,action))
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




