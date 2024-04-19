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
        self.transition = np.array([[ 5.09781492e-01, 1.00818377e+00,-2.13421761e-04, 8.11366893e-04,
  2.93404165e-04, 4.33318792e-04,-6.56777134e-05, 9.42977155e-04,
  7.43520151e-04,-3.64902225e-05, 9.80097126e-04,-6.23083955e-01,
  -2.31987643e-02,-4.37032353e-02,-2.36831534e-02,-3.23232915e-02,
  -4.47997378e-02,-3.59208547e-02, 9.93481418e-03,-1.58870163e-02,
  -4.50122942e-02],
 [ 5.98606690e-01, 8.06386768e-04, 1.00717500e+00, 1.96795223e-04,
  3.00610432e-04, 1.91905026e-03, 4.86357900e-04, 8.43128727e-04,
  3.07764702e-04,-1.13799134e-03, 1.02118916e-03,-4.02784818e-03,
  -5.90966089e-01, 1.17972777e-02,-3.54311246e-02,-2.72612441e-03,
  1.89676986e-02,-1.61464021e-02,-1.27071331e-02,-2.24772701e-02,
  3.58142112e-02],
 [ 5.59518991e-01, 1.31884932e-03, 1.64721984e-04, 1.00695323e+00,
  1.12697735e-05,-1.56434387e-03, 1.20640619e-04,-3.91408351e-04,
  -6.54681223e-04, 1.81562120e-03,-6.50738652e-04, 2.19305959e-02,
  -2.59844799e-02,-5.57979750e-01,-1.21347698e-02, 1.30110108e-03,
  -5.47076604e-04, 1.40404536e-02,-2.08586086e-02,-1.70985220e-02,
  -2.71143472e-02],
 [ 5.90371645e-01,-3.51139269e-04, 1.19088644e-03, 2.43635495e-03,
  1.00536674e+00, 1.97112414e-03, 1.12235111e-03,-1.35949159e-03,
  1.58908290e-03, 4.03789341e-04, 9.90508190e-04,-3.45964531e-02,
  2.24451240e-02,-4.28284393e-03,-5.62033069e-01,-9.93200621e-04,
  -6.18007584e-02, 3.94991800e-02,-4.66859215e-02,-1.43806106e-03,
  -2.72715059e-02],
 [ 5.70543335e-01,-3.92665693e-04,-5.18968365e-05,-3.50939065e-04,
  9.43487654e-04, 1.00440261e+00, 1.66734298e-04, 5.48378220e-04,
  -1.89490592e-03, 1.58545598e-03, 1.10776382e-04, 1.14393458e-01,
  -1.78562261e-02,-1.14459329e-02,-9.20791567e-03,-5.61305494e-01,
  -1.04897067e-01,-7.85300392e-03,-1.22262639e-03,-4.52522846e-02,
  -5.41019465e-02],
 [ 5.79644143e-01,-8.37033654e-05,-4.43929600e-04,-1.48330909e-04,
  1.13845504e-03,-5.03963773e-05, 1.00805530e+00, 3.98014986e-04,
  1.57627147e-04,-3.60913372e-04,-3.20515818e-04,-2.71301258e-03,
  -1.75614196e-02,-4.47135553e-02,-5.84055096e-02,-4.79843470e-02,
  -5.46979751e-01,-5.48082427e-02,-4.02750689e-02,-2.34499567e-02,
  -1.72967302e-02],
 [ 5.57834525e-01,-2.23442453e-04,-5.06627022e-04, 1.12939682e-04,
  -1.01358400e-03,-9.70215286e-04,-7.02776793e-04, 1.00490419e+00,
  4.02802534e-04, 7.31655753e-04, 1.23319084e-03,-1.66290325e-02,
  2.11764794e-02, 1.04253238e-02,-1.53898893e-02,-3.29362082e-02,
  -5.28153109e-02,-5.38715943e-01,-8.26084923e-03,-2.70311987e-02,
  1.42591196e-02],
 [ 5.75601917e-01,-3.38821121e-04, 2.96692733e-04, 7.39846450e-04,
  -2.59045073e-04, 4.12116907e-04, 2.30788618e-03,-1.27065805e-03,
  1.00757410e+00,-8.62445589e-04, 3.66182507e-05, 7.30810755e-03,
  -7.95801231e-03,-1.57510814e-02,-4.46505504e-02,-8.30841974e-03,
  -2.27723371e-02,-1.81534300e-02,-5.66735025e-01,-1.13859351e-02,
  -2.05012364e-02],
 [ 4.54834177e-01, 4.36142331e-04, 6.59195757e-04, 1.67358859e-03,
  -1.20789168e-04, 1.11784604e-03,-3.38335067e-04,-5.57824438e-04,
  8.07813586e-05, 1.00506557e+00, 4.75891122e-04,-4.23514010e-02,
  -8.32047873e-04,-8.86192705e-04, 1.14801522e-02, 5.78317453e-03,
  -6.57861549e-02,-1.77269669e-02,-1.93323613e-02,-5.55346735e-01,
  -6.39938634e-02],
 [ 5.72681757e-01, 7.47161522e-04,-1.18592772e-04,-1.76632821e-04,
  1.02087703e-04, 1.13169651e-03, 1.14722695e-03, 1.14489563e-03,
  -2.72900756e-04,-5.26368027e-04, 1.00802491e+00,-1.02538356e-01,
  -3.51472908e-02,-4.45970027e-02,-1.54292909e-02,-2.36021422e-02,
  -8.01297525e-02, 2.28440201e-02,-6.15383148e-02,-1.57406706e-02,
  -5.82427325e-01]])
  
        
#         self.transition = np.array([[ 5.60952304e-01, 1.00740161e+00, 7.81902690e-04,-8.06913867e-04,
#   2.37281928e-04, 1.09335371e-03, 6.76221033e-04, 1.16729686e-03,
#   -4.63787327e-04, 1.25786128e-03, 1.66899899e-03,-5.88019402e-01,
#   -3.61340267e-02,-2.67765555e-02,-3.19773658e-02,-3.04568960e-02,
#   -7.02861383e-02,-6.18205400e-02,-2.68295438e-02,-1.75349739e-02,
#   -4.20531838e-02],
#  [ 5.33236139e-01, 2.91675164e-04, 1.00546293e+00,-6.63261911e-04,
#   1.69511513e-03,-1.49764581e-04,-1.53459245e-04,-1.10133951e-03,
#   3.69939718e-04,-1.67432336e-04,-2.01386312e-03,-1.47443964e-02,
#   -5.57347530e-01,-1.13178984e-02,-1.31893000e-02, 9.79232905e-03,
#   1.66240568e-02,-6.58975143e-04,-3.16453077e-03,-1.11762606e-02,
#   4.06601074e-02],
#  [ 5.16310826e-01, 1.18613549e-03, 7.14435933e-04, 1.00485932e+00,
#   -6.09806355e-04, 1.97287228e-03, 1.88068056e-03, 8.59806655e-04,
#   -4.86946864e-04, 5.33290163e-05,-1.87565482e-04, 1.13409988e-02,
#   -1.79078620e-03,-5.61824600e-01, 9.34669916e-04,-2.16725955e-02,
#   -5.95707543e-03, 1.13523236e-02, 2.27427422e-02,-2.11364559e-02,
#   -2.20277338e-02],
#  [ 5.22365541e-01,-5.99692451e-04, 1.17920259e-05,-1.09904307e-03,
#   1.00546791e+00,-1.26852284e-03, 8.31058567e-04, 1.36407404e-04,
#   -1.83373445e-04,-1.30882921e-04,-6.68427547e-04,-1.37753497e-02,
#   2.52994264e-02, 4.73655544e-02,-5.93547538e-01, 2.96535865e-03,
#   -5.94436412e-02, 3.43896075e-02, 2.65001583e-03,-1.89525530e-02,
#   -9.23482780e-03],
#  [ 5.25060468e-01,-3.95352151e-04,-1.17620847e-03,-1.71764961e-04,
#   -9.37447848e-04, 1.00519692e+00,-4.39580799e-04,-8.51038541e-04,
#   -7.37664747e-04, 7.53087637e-07, 1.19712611e-03, 1.29885057e-01,
#   7.59584123e-03, 4.25724048e-03,-4.83577271e-02,-5.51201283e-01,
#   -9.81643343e-02,-5.98110089e-03,-1.10867429e-02,-3.27926929e-02,
#   -1.89642687e-02],
#  [ 5.40816770e-01, 4.93928053e-04, 4.22704358e-05,-8.59393730e-04,
#   2.67535714e-03,-1.72697730e-03, 1.00564499e+00,-7.14343977e-04,
#   -6.10257830e-04, 2.61568632e-03, 9.32339969e-05,-3.97034327e-02,
#   1.35762822e-02,-4.91302019e-02,-2.00190353e-02,-3.76503490e-02,
#   -5.60906428e-01,-2.49201746e-02,-1.65993375e-03,-1.32333920e-02,
#   -5.94331815e-02],
#  [ 5.12987536e-01,-8.97161061e-04, 3.86069070e-04,-6.60642565e-04,
#   5.11031534e-05, 1.24209444e-04, 8.19331029e-04, 1.00573613e+00,
#   -8.66240472e-04, 9.32733675e-04, 1.43179714e-04,-4.40523306e-02,
#   2.54875279e-02, 5.72426005e-03,-5.78259176e-03,-9.16675913e-03,
#   -2.80173408e-02,-5.10688819e-01,-1.16815509e-02,-2.34385061e-02,
#   -1.71670259e-02],
#  [ 5.20322799e-01, 8.39645409e-04, 1.69083555e-03,-1.42799542e-03,
#   1.62826414e-03, 8.05921857e-04, 1.12237359e-03, 6.35996441e-04,
#   1.00615084e+00, 1.83799446e-03, 1.29525475e-03, 1.52830716e-02,
#   -3.68982631e-02, 2.50020006e-02,-4.49035678e-02, 3.38695918e-02,
#   -2.27226338e-02,-2.60774973e-02,-5.39753381e-01,-2.75251692e-02,
#   -6.98547480e-04],
#  [ 5.53084974e-01, 9.77105728e-04,-1.25392099e-03, 8.36609670e-04,
#   6.70791256e-04, 9.96408209e-04,-5.07732072e-04, 8.90634439e-04,
#   -8.05852877e-04, 1.00773206e+00, 1.13022174e-04,-4.17901191e-02,
#   2.19111005e-02,-1.01315152e-02,-1.20132521e-02,-2.03225691e-02,
#   -7.57687985e-02,-3.06919387e-02,-2.35220958e-02,-5.77316478e-01,
#   -8.67566471e-02],
#  [ 5.06594489e-01, 3.88713237e-04,-8.27805268e-04, 6.54069345e-04,
#   -6.55077429e-04,-2.26885400e-04, 7.49557014e-04,-1.01846469e-03,
#   1.61930688e-03,-1.20832514e-03, 1.00796964e+00,-1.14602740e-01,
#   -5.99690634e-02, 4.74409637e-03,-2.46268659e-03, 2.01195222e-02,
#   -8.48901859e-02, 1.14227495e-02,-2.85144059e-02,-1.85221624e-02,
#   -5.78959113e-01]])
        

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




