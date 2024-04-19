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
#         self.transition = np.array([[-1.36539352e-03, 5.01593719e-01, 2.01468186e-01, 7.10466547e-04,
#   2.97313380e-03,-1.21560974e-03,-3.44091932e-03, 5.13559027e-04,
#   4.01616797e-04, 5.22623112e-03,-1.79144152e-03, 5.20872169e-02,
#   -7.03173118e-03,-1.28071196e-02, 3.70637126e-03,-1.69490821e-03,
#   -2.43384238e-03, 1.17656686e-03, 6.83136765e-05,-3.87757068e-03,
#   -1.02619144e-03],
#  [ 1.96533976e-03, 2.00559459e-01, 4.98294504e-01, 1.96744514e-01,
#   4.83019715e-03,-5.90844881e-03, 4.50176160e-04,-2.78497701e-03,
#   2.42311473e-03, 1.74323206e-03,-1.33719421e-03,-8.92954832e-03,
#   3.62873140e-02,-6.25090369e-03,-1.00825857e-02, 6.26767285e-03,
#   -1.92189224e-03,-1.59182918e-03, 3.45868348e-03,-5.83719041e-03,
#   -4.04190161e-03],
#  [ 1.38220700e-03, 3.13206188e-03, 1.92997291e-01, 5.00936649e-01,
#   2.04158619e-01,-1.90551919e-03, 3.62256775e-04, 2.56704070e-03,
#   -1.61225685e-03,-2.32114939e-03, 4.24624868e-03,-1.89512544e-02,
#   -1.63802522e-03, 3.48088717e-02, 1.21199129e-04,-1.66614977e-02,
#   -2.26882118e-03, 1.28348372e-03, 3.61444564e-03, 2.40779795e-03,
#   -4.37217070e-03],
#  [ 1.64932080e-03,-1.03824436e-03, 1.18827862e-03, 1.96957219e-01,
#   5.00789016e-01, 1.99133921e-01, 1.09027264e-03, 9.39012767e-04,
#   -4.40447224e-03, 5.11947109e-03, 3.19217631e-04,-3.09920854e-03,
#   -1.60499714e-02, 1.84749655e-04, 3.56093308e-02,-6.70508178e-04,
#   -8.36297118e-03,-2.59071548e-03,-6.20203843e-03, 3.85494313e-04,
#   -1.54953338e-03],
#  [ 3.93649180e-03, 5.49499645e-03,-1.80933059e-04,-1.49198879e-03,
#   1.95964285e-01, 5.07616580e-01, 1.93838057e-01,-3.59055738e-04,
#   -2.28271132e-03, 4.35516350e-03,-4.05695282e-03,-3.77600474e-03,
#   2.56842167e-04,-1.52173094e-02,-3.94606721e-03, 3.05834882e-02,
#   -9.70406476e-04,-1.33986448e-02, 2.57291024e-03,-1.71535255e-04,
#   6.36142789e-04],
#  [-1.70222135e-03, 2.99734577e-03,-2.29907814e-04, 3.39605606e-03,
#   -6.34711864e-03, 2.02495594e-01, 5.00353194e-01, 1.97251791e-01,
#   -1.62368774e-03, 2.03325458e-03,-1.22416550e-03, 4.70187871e-03,
#   -5.38557380e-03, 6.92245683e-03,-1.38343520e-02,-3.64022308e-03,
#   3.16350826e-02,-7.97902800e-03,-9.28773919e-03,-3.14138516e-03,
#   4.10327995e-03],
#  [ 1.84157369e-03, 3.02004784e-03, 1.99555771e-03,-7.35830973e-04,
#   -5.83100883e-04,-1.35992540e-03, 2.02846707e-01, 4.94348669e-01,
#   2.00941154e-01,-1.54741404e-03, 4.66660452e-03,-2.82143910e-03,
#   -3.70894493e-03,-2.05605382e-03,-1.47773924e-03,-1.27512819e-02,
#   -3.83819574e-03, 3.65625206e-02,-2.46456302e-03,-1.25197324e-02,
#   2.77975902e-03],
#  [ 6.68155555e-04,-1.37039359e-03, 1.34563186e-03,-1.36577920e-03,
#   -2.30700776e-03, 1.17907611e-03,-1.83036777e-03, 2.04494978e-01,
#   5.00408273e-01, 1.97828566e-01, 2.81772993e-05,-2.95492901e-03,
#   -5.03013587e-03,-5.97559580e-04, 3.87640467e-03,-6.22167203e-04,
#   -2.00601042e-03,-6.25740799e-03, 3.35155953e-02,-2.98739764e-03,
#   -1.38045153e-02],
#  [-4.17699829e-03,-3.04046996e-03,-8.39238520e-04,-6.14881451e-03,
#   1.88757070e-03,-3.98637795e-03, 2.63378483e-03, 1.73838424e-03,
#   2.03034429e-01, 4.96279961e-01, 2.01792985e-01, 5.70097892e-03,
#   -7.67705155e-03,-4.61116064e-04, 6.00649762e-04, 1.29172758e-03,
#   8.68908441e-03,-1.57863965e-02, 1.13644578e-03, 4.04409042e-02,
#   -5.43528726e-03],
#  [-2.05547370e-03,-2.29414075e-03, 3.82301926e-03,-1.04188830e-02,
#   5.80020012e-03,-3.92973033e-03, 9.89776031e-04,-1.11689277e-03,
#   4.23222833e-03, 1.99304453e-01, 4.99408730e-01,-5.59583456e-04,
#   3.24290142e-03, 5.02017771e-04, 2.16459971e-03, 1.37183093e-03,
#   -7.09125627e-04,-2.52243190e-03,-8.19228858e-03, 1.97421453e-03,
#   5.02865867e-02]])


        self.transition = np.array([[ 3.14762336e-03, 4.96940769e-01, 2.02338369e-01, 1.07977051e-03,
  -2.22181157e-03,-8.66009289e-04,-4.28625683e-04,-1.24461472e-03,
   7.04701298e-03,-6.19907902e-03, 4.94953179e-04, 5.03493902e-02,
  -3.24898972e-03,-1.26755047e-02,-2.54553431e-04,-4.53224382e-03,
   4.72013371e-03,-3.53322437e-03, 4.76449581e-03,-2.43580642e-03,
  -2.72414217e-04],
 [ 3.00158010e-03, 1.99807484e-01, 5.00888143e-01, 1.97261609e-01,
   3.54107017e-03,-2.35498546e-03, 3.25982452e-03, 2.03114446e-04,
  -4.17097448e-03, 5.55128937e-03,-5.80744032e-04,-2.51182764e-03,
   3.42416599e-02,-5.43024218e-03,-1.29222453e-02,-9.48685623e-03,
   2.59719140e-03, 7.07992104e-04,-2.03051175e-03,-1.37069174e-03,
  -6.40014187e-04],
 [ 3.91797419e-03, 6.71623432e-04, 1.95643617e-01, 4.98058037e-01,
   1.98040648e-01, 4.91956948e-03,-2.86508558e-03, 3.09259844e-03,
  -2.13152763e-04, 2.08631905e-03, 4.92526533e-03,-9.09430340e-03,
  -6.65019162e-03, 3.23643535e-02, 3.72957514e-03,-2.20391663e-02,
   1.10579182e-03, 1.07580424e-03, 9.79060132e-04,-2.47640038e-03,
   5.52685659e-03],
 [-9.82303047e-04,-3.59077595e-03,-1.50258987e-03, 2.03274970e-01,
   4.98916861e-01, 2.05324956e-01,-7.96504545e-03, 1.56322352e-03,
  -4.54733664e-04, 4.48604690e-03,-1.88831175e-03, 6.53176087e-03,
  -1.39507069e-02,-5.89624077e-03, 3.24807438e-02,-7.12216968e-03,
  -1.16465584e-02, 1.13291032e-02,-1.85803711e-03, 2.61319981e-03,
   9.06290571e-04],
 [-1.23940928e-03, 5.28917922e-03, 8.97077576e-04,-1.75428143e-03,
   2.04170969e-01, 4.96537245e-01, 2.01772632e-01,-3.95998147e-03,
   2.41862486e-03, 3.70302021e-03,-4.58446430e-03, 1.07750898e-03,
   3.55907501e-03,-1.86900148e-02,-6.72687418e-03, 3.24774650e-02,
   3.68230886e-03,-1.01702107e-02,-3.21889516e-03,-5.30132698e-03,
   3.32599448e-03],
 [ 2.42489513e-04, 2.79122826e-03, 3.27295589e-03, 5.68643571e-04,
  -1.69867987e-03, 2.05234376e-01, 4.94923337e-01, 1.98337493e-01,
   4.24493902e-03,-1.71480091e-03,-2.44276831e-03,-6.35311578e-03,
   1.75853033e-03,-2.54793114e-03,-1.14611361e-02,-2.33622873e-03,
   3.21738171e-02,-9.09293677e-03,-1.31748491e-02,-3.44062983e-03,
   9.24996255e-04],
 [-3.59594780e-03,-1.15580530e-03, 3.86974043e-03,-3.35224744e-03,
   3.80783613e-03, 3.00331998e-04, 1.96590025e-01, 5.03004841e-01,
   1.94308750e-01, 7.74875422e-04,-3.33614753e-03,-7.54691418e-03,
   1.52686104e-04,-5.04033023e-03, 6.81904424e-03,-1.30402617e-02,
  -3.86507046e-03, 3.93726997e-02,-3.50593541e-03,-1.13910337e-02,
   1.99904398e-03],
 [ 2.25215953e-03, 9.20867764e-04,-2.50806280e-03, 4.36846404e-03,
  -5.32387903e-03, 3.98606777e-03,-3.02859973e-03, 1.96433709e-01,
   4.99503961e-01, 1.98926122e-01,-5.55716993e-04,-3.37991249e-03,
  -2.30348393e-05,-1.32044116e-03, 1.18234648e-03,-1.53772997e-03,
  -9.05211070e-03,-3.71845683e-03, 3.34591325e-02,-1.05681569e-03,
  -8.21635793e-03],
 [ 1.57574707e-03, 7.80888897e-03,-4.67979920e-03, 2.83953076e-03,
  -2.18868958e-03, 9.64513661e-03,-5.73136964e-03,-4.27404423e-03,
   2.00749134e-01, 4.99660009e-01, 1.99382486e-01,-3.17500058e-03,
  -8.13851180e-05, 2.01957171e-03,-2.00370810e-04, 5.09862533e-03,
   2.73560300e-03,-1.47805074e-02,-1.04153524e-02, 3.81305868e-02,
  -8.21024320e-04],
 [ 6.69463204e-03, 3.10628552e-03,-2.49726627e-03, 2.65207778e-03,
  -4.97830914e-04, 2.16250752e-03,-6.37845193e-03, 1.13130454e-03,
   2.66452050e-04, 1.98966335e-01, 5.00932405e-01,-1.57985761e-03,
   7.90995299e-04, 7.36500770e-04,-1.84470088e-03, 6.83469120e-06,
  -4.44214999e-03, 4.90861501e-03,-1.49738578e-02,-1.17558619e-02,
   5.33643512e-02]])
    

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




