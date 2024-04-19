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
#         self.transition = np.array([[-1.31799777e-03, 4.91788747e-01, 2.13163220e-01,-4.02622920e-03,
#   -3.50365139e-03,-2.49510727e-03, 4.74394004e-03, 4.92743663e-03,
#   1.09703361e-03, 2.09009324e-03, 6.69812182e-03,-3.55537128e-01,
#   -3.79247029e-02, 5.31647952e-03, 2.51955518e-02, 3.27154161e-02,
#   2.87293850e-02, 3.17495548e-02, 2.67047711e-02, 1.11741846e-02,
#   1.94772368e-02],
#  [ 2.14982747e-03, 1.93683525e-01, 4.94287490e-01, 1.90524420e-01,
#   3.04166546e-03, 6.17783801e-03, 6.62876751e-03, 2.99702049e-03,
#   -5.08950848e-03, 4.62704580e-03, 4.36052428e-03,-5.16591428e-02,
#   -3.39122867e-01,-6.31961609e-02, 5.20363884e-03, 1.65219684e-02,
#   2.87450644e-02, 2.50058176e-02, 2.36550186e-02, 2.09306857e-02,
#   2.07329797e-02],
#  [-7.28771733e-03, 1.17609693e-03, 1.95931606e-01, 5.01323193e-01,
#   2.07001400e-01, 1.21663836e-02,-1.18126968e-03, 1.36829216e-02,
#   1.15135930e-03,-5.94860115e-03,-3.28255846e-03, 1.87453597e-02,
#   -4.83369925e-02,-3.56105242e-01,-4.70221472e-02, 2.24205269e-02,
#   3.05832026e-02, 3.55677416e-02, 2.41518652e-02, 3.24789944e-02,
#   2.98884136e-02],
#  [-1.53235315e-02,-2.20309625e-03, 2.20031323e-03, 2.01873555e-01,
#   4.94693639e-01, 2.04292216e-01,-2.15094065e-03,-1.49800611e-03,
#   -9.15875045e-03, 9.31411096e-04,-2.08972865e-03, 1.91260634e-02,
#   1.53448321e-02,-5.58542153e-02,-3.44803852e-01,-3.38449861e-02,
#   2.52864598e-02, 2.50391053e-02, 1.07616898e-02, 2.85467056e-02,
#   3.51398069e-02],
#  [-9.89590148e-03, 4.69015823e-03, 3.83470331e-03,-3.96861906e-03,
#   1.94231312e-01, 5.08677319e-01, 2.02633193e-01,-1.77416161e-03,
#   3.88373756e-03, 1.18509591e-02, 9.36773043e-04, 2.48799400e-02,
#   1.13367350e-02, 1.30426771e-02,-5.79672857e-02,-3.48478747e-01,
#   -4.69563651e-02, 2.66841035e-02, 3.01627196e-03, 1.15660132e-02,
#   1.02727818e-02],
#  [-9.80793294e-03, 9.82602365e-03, 4.02030318e-03, 6.38025808e-04,
#   -2.42003222e-03, 2.12858858e-01, 4.91087379e-01, 2.04980780e-01,
#   9.67159543e-03,-1.84989920e-03, 5.41346885e-04, 2.24084093e-02,
#   2.20581081e-02, 1.48585909e-02, 1.74117328e-02,-5.78103096e-02,
#   -3.52530232e-01,-6.16635241e-02, 9.92951470e-03, 9.16607296e-03,
#   2.10155528e-02],
#  [ 2.22876946e-03, 3.10975911e-04,-1.63222309e-03,-1.52579977e-02,
#   1.05840391e-03,-3.11064437e-03, 1.93610505e-01, 5.03393624e-01,
#   1.99226576e-01,-7.40712270e-04,-3.97534077e-03, 1.51251964e-02,
#   3.16486468e-02, 2.62377668e-02, 2.00050059e-02, 1.86375248e-02,
#   -5.48381041e-02,-3.60405619e-01,-4.83343457e-02, 1.62829166e-02,
#   1.41802834e-02],
#  [ 1.06856580e-02, 5.20579977e-03,-1.13411800e-02,-9.11416305e-03,
#   -2.43858149e-03,-3.15824224e-03,-2.43724204e-03, 1.93636374e-01,
#   5.00191953e-01, 2.05132425e-01, 1.44873794e-03, 2.16488976e-02,
#   1.86552254e-02, 2.57748714e-02, 2.72550259e-02, 2.79418165e-02,
#   1.04149840e-02,-5.35865161e-02,-3.48371157e-01,-5.25234211e-02,
#   1.84499488e-02],
#  [ 1.32421275e-03,-4.06949596e-03,-5.50118793e-03, 1.03807568e-02,
#   -3.63113387e-03,-1.11008410e-02,-4.61043263e-03, 3.71555074e-03,
#   1.93569064e-01, 5.04759197e-01, 2.00840637e-01, 1.14734562e-02,
#   1.16343706e-02, 3.81062204e-02, 1.80092646e-02, 2.23675793e-02,
#   2.78482707e-02, 2.59051247e-02,-5.72909018e-02,-3.59925907e-01,
#   -5.18313821e-02],
#  [ 3.50258904e-03, 8.29833526e-03,-9.65048379e-03,-2.58235527e-03,
#   -2.27404208e-03, 9.00994332e-03, 9.22543305e-03, 1.71804830e-03,
#   -2.59352910e-03, 2.04432870e-01, 4.91417341e-01, 1.29053167e-02,
#   1.80442548e-02, 2.56664748e-02, 2.63702023e-02, 1.70848449e-02,
#   2.19154779e-02, 3.13675821e-02, 1.60466141e-02,-5.91185482e-02,
#   -3.48162009e-01]])
    
    
        self.transition = np.array([[-3.23280219e-04, 4.86171323e-01, 1.98423038e-01,-1.86831836e-03,
  -3.95641011e-03, 1.11110401e-02, 1.81000887e-03, 2.65046518e-03,
   4.12617625e-03, 5.40443086e-03,-5.79622962e-04,-3.46915103e-01,
  -5.60749631e-02, 2.36803728e-02, 1.82234790e-02, 2.06401937e-02,
   6.96310669e-03, 1.28573999e-02, 2.29551611e-02, 1.51491934e-02,
   2.34269484e-02],
 [ 3.43581373e-04, 2.00804640e-01, 4.91247194e-01, 2.01208535e-01,
  -2.08178057e-03,-6.59742296e-03, 1.06466786e-02, 1.23743930e-02,
   1.00450546e-02,-5.34257406e-03,-5.47626182e-04,-4.81851624e-02,
  -3.60950361e-01,-5.48219209e-02, 7.46345164e-03, 1.44317052e-02,
   1.77519513e-02, 1.17990833e-02, 2.48609731e-02, 8.59600290e-03,
   2.22053399e-02],
 [-2.75621013e-03, 1.22993178e-03, 1.88003996e-01, 4.94835805e-01,
   2.03510968e-01,-1.72383343e-03, 1.39610741e-03, 8.44763323e-03,
  -7.46558008e-03, 3.90783334e-03,-6.33990961e-03, 1.88220393e-02,
  -4.87696216e-02,-3.72306234e-01,-5.68717218e-02, 1.53434734e-02,
   1.63876317e-02, 1.99050215e-02, 2.22818605e-02, 9.13253803e-03,
   1.48773812e-02],
 [-9.11644104e-03,-5.31724914e-03, 4.38116482e-03, 2.03226594e-01,
   4.94874905e-01, 2.00731418e-01, 9.42673776e-03,-3.04560851e-03,
  -5.23381350e-04, 4.03440258e-03, 3.14780046e-03, 3.42982794e-02,
   2.15517263e-02,-5.70263008e-02,-3.65910120e-01,-4.37715427e-02,
   1.42105824e-02, 1.42947319e-02, 2.66454670e-02, 6.74227336e-03,
   1.27416484e-02],
 [-8.09868003e-03, 1.45915949e-03, 9.48205008e-03,-1.94249265e-03,
   2.05444140e-01, 5.06170810e-01, 2.17725519e-01,-5.57423278e-03,
  -3.39549795e-03, 9.91626596e-03, 2.84230953e-03, 3.12225813e-02,
   1.35098258e-02, 2.08407927e-02,-4.35549953e-02,-3.54058918e-01,
  -5.67729496e-02, 1.98146448e-02, 2.94958168e-02, 2.41840206e-02,
   1.05591936e-02],
 [ 2.46661154e-04,-1.58518762e-03,-3.40223954e-03,-7.58715089e-03,
   9.93955807e-03, 2.01459619e-01, 5.05915990e-01, 2.10654024e-01,
  -1.00115216e-02, 4.93838461e-03, 3.42588806e-03, 3.97183172e-02,
   1.69001823e-02, 2.12497836e-02, 2.51419087e-02,-4.97361729e-02,
  -3.66050452e-01,-5.16394893e-02, 1.58226303e-02, 2.97224274e-02,
   2.25799213e-02],
 [-2.37104066e-02, 6.25383417e-03, 9.79921309e-03,-1.53599565e-03,
   1.23261586e-02,-6.14580491e-05, 1.91598441e-01, 4.89140712e-01,
   1.93218769e-01, 2.95689640e-03, 8.29772924e-03, 2.73772652e-02,
   1.75978484e-02, 1.39869525e-02, 2.29353157e-02, 1.41944615e-02,
  -6.58978556e-02,-3.63537898e-01,-4.93202895e-02, 2.05192275e-02,
   2.09373068e-02],
 [-1.25957167e-02, 6.53653460e-03, 7.41878955e-03, 1.82448424e-03,
   6.62029274e-03,-8.43920715e-03,-3.11210217e-03, 1.99703678e-01,
   4.97076284e-01, 1.94436038e-01, 5.71151974e-03, 1.81654423e-02,
   1.30767307e-02, 2.03820585e-02, 2.08783731e-02, 2.66079356e-02,
   1.27522084e-02,-4.18452334e-02,-3.56585192e-01,-7.10956435e-02,
   1.50685792e-02],
 [-1.50104106e-02, 8.27885783e-03,-1.04365755e-03, 4.18220819e-03,
  -7.34886009e-03, 4.60717938e-04,-1.01823446e-03,-5.42415509e-03,
   2.01476772e-01, 4.96085300e-01, 1.94349449e-01, 1.91638784e-02,
   1.31956800e-02, 2.84545170e-02, 3.24388013e-02, 3.34600246e-02,
   1.39077703e-02, 2.53526818e-02,-5.45985662e-02,-3.58123467e-01,
  -4.50182798e-02],
 [-5.55341105e-03, 1.27396355e-02,-2.43014024e-03, 1.26350922e-02,
  -2.30198374e-03, 2.71830281e-03, 1.31669466e-03, 3.58228879e-03,
   7.66068997e-03, 1.96816170e-01, 4.93915517e-01, 3.04090474e-02,
   2.52578574e-02, 3.36951270e-02, 4.01135078e-02, 2.61388855e-02,
   1.27534998e-02, 3.87912433e-02,-1.56743575e-03,-4.81968253e-02,
  -3.43594267e-01]])

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




