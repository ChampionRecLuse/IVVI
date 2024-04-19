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
        self.transition = np.array([[ 1.54859214e-03, 1.01720975e+00,-2.12571225e-03,-2.21621160e-04,
  -1.02475340e-02,-5.85112629e-03, 5.94883714e-03, 1.01933904e-03,
  -9.94377905e-03,-7.36330189e-03,-1.09861058e-02, 3.91770840e-02,
  3.86788931e-02, 3.61410189e-02, 9.16257567e-02, 8.97216894e-02,
  8.43714531e-02, 8.89741476e-02, 8.89307718e-02, 9.18300225e-02,
  8.99103630e-02],
 [ 4.96118038e-02, 9.56265509e-03, 9.93609323e-01, 7.05876555e-03,
  -5.35061942e-03,-1.07995287e-02, 3.20858935e-03, 6.26992310e-03,
  -1.09658677e-03,-1.79232299e-03, 2.10117209e-04, 7.75046609e-02,
  8.10387444e-02, 7.61776276e-02, 1.08597367e-01, 1.12983703e-01,
  1.12743559e-01, 1.16132595e-01, 1.15768608e-01, 1.12350550e-01,
  1.23510770e-01],
 [ 4.77679237e-02, 2.47381136e-02,-9.75980267e-03, 1.00015097e+00,
  -2.25890169e-03,-1.54485918e-02, 2.78815361e-04, 3.59966962e-03,
  -1.69711756e-03,-1.17131031e-04,-1.00086111e-02, 7.62854076e-02,
  7.15322063e-02, 7.88278309e-02, 1.14848635e-01, 1.16422692e-01,
  1.15065768e-01, 1.17995084e-01, 1.14999104e-01, 1.17980758e-01,
  1.08297076e-01],
 [ 4.13711959e-02, 2.15071745e-02,-2.59921456e-03, 9.31502689e-03,
  9.89612876e-01,-1.05897708e-02,-1.67743835e-03,-2.15002145e-03,
  -3.88845020e-03,-8.80780258e-03,-6.17746336e-03, 6.76790856e-02,
  6.87623747e-02, 6.98484098e-02, 1.18517145e-01, 1.07731989e-01,
  1.10379943e-01, 1.05424809e-01, 1.10686597e-01, 1.06221338e-01,
  1.16599815e-01],
 [ 5.32240011e-02, 2.27946597e-02,-7.22931702e-03, 7.49404173e-03,
  -1.72244241e-02, 9.87542120e-01,-4.11141685e-04, 4.74525739e-04,
  6.71209006e-04,-1.32533591e-02,-5.80899664e-03, 8.25226833e-02,
  8.25683083e-02, 8.49080767e-02, 1.13319242e-01, 1.17022017e-01,
  1.07754025e-01, 1.22098013e-01, 1.15286733e-01, 1.18118057e-01,
  1.15417716e-01],
 [ 1.15898634e-02, 1.69823789e-02,-2.78502166e-03, 2.81446026e-03,
  -6.49126439e-03,-1.33101936e-02, 9.99937015e-01, 5.35079894e-03,
  -9.78805980e-03,-1.77042895e-03,-7.06323555e-05, 3.58281397e-02,
  4.40757826e-02, 4.20694353e-02, 1.01874800e-01, 8.71569010e-02,
  1.10104794e-01, 9.66108386e-02, 9.55741485e-02, 9.84170985e-02,
  9.82372205e-02],
 [ 1.98309570e-02, 1.11337193e-02,-8.33490527e-03, 2.01771926e-03,
  6.92036279e-04,-7.66733745e-03, 1.15964784e-03, 1.00277488e+00,
  6.24911868e-03,-1.22006170e-02,-1.65567356e-02, 5.39166088e-02,
  5.66819392e-02, 5.23138539e-02, 9.03150614e-02, 9.84711053e-02,
  9.45778431e-02, 1.07577902e-01, 1.00627119e-01, 9.84436231e-02,
  1.03758015e-01],
 [ 3.90383439e-02, 1.22984501e-02,-1.48634032e-02, 1.05356914e-02,
  3.35772533e-03,-1.26052542e-02, 4.21985656e-03, 6.35725693e-03,
  1.00166791e+00,-9.27826524e-03,-7.59896923e-03, 7.15082850e-02,
  6.59898694e-02, 7.10208328e-02, 1.06064321e-01, 1.02441582e-01,
  1.08384208e-01, 1.08455053e-01, 1.16011587e-01, 1.10087691e-01,
  1.14079730e-01],
 [ 8.64070236e-03, 1.55492295e-02,-1.27559902e-02, 8.27428320e-03,
  4.52642760e-03,-1.81420296e-02, 2.44456968e-03,-1.56154041e-03,
  7.43288310e-03, 9.93222690e-01,-1.59463239e-02, 4.29592762e-02,
  4.20181592e-02, 4.73544080e-02, 9.06532750e-02, 8.67697521e-02,
  9.20444173e-02, 9.13777538e-02, 9.62424998e-02, 9.99282797e-02,
  9.48821962e-02],
 [ 7.20730046e-03, 2.32483908e-02,-1.22687571e-02, 1.40295898e-03,
  -4.48601841e-03,-1.81137577e-02, 3.68576395e-04,-2.38432159e-03,
  -3.08304914e-03,-4.66746280e-03, 9.89943734e-01, 3.61001034e-02,
  4.16727835e-02, 4.09450733e-02, 9.19085590e-02, 9.19074256e-02,
  8.94972723e-02, 9.86535022e-02, 8.82444251e-02, 9.27970455e-02,
  1.08298513e-01]])
        
#         self.transition = np.array([[ 1.03125594e-02, 1.00278831e+00,-2.08546662e-02,-2.12654395e-02,
#   -9.96920868e-03,-8.49501168e-03, 5.14357939e-03,-4.68881937e-03,
#   -2.42966647e-03,-8.92829867e-03, 5.45382785e-03, 2.79227119e-02,
#   3.16394968e-02, 2.98124495e-02, 7.72575580e-02, 8.53261107e-02,
#   8.79745650e-02, 1.07964611e-01, 1.03193591e-01, 8.83490149e-02,
#   1.08972542e-01],
#  [ 4.76217416e-02, 1.24418916e-02, 9.85291798e-01,-1.12014066e-02,
#   -2.34340195e-03,-1.78254362e-03,-2.50890594e-03,-3.95220335e-03,
#   -1.23732983e-03, 9.85353647e-04, 9.51352941e-03, 6.58206475e-02,
#   6.55851646e-02, 6.55177025e-02, 1.12468393e-01, 1.15139127e-01,
#   1.15468747e-01, 1.15195859e-01, 1.12002754e-01, 1.01525413e-01,
#   1.19782404e-01],
#  [ 3.11099988e-02, 1.45238902e-02,-6.57126123e-03, 9.99442172e-01,
#   -8.68595185e-03,-1.36130536e-02,-3.42959153e-03,-7.09225328e-03,
#   5.67920210e-03, 3.40672432e-03, 6.83053143e-03, 5.27323644e-02,
#   4.42041107e-02, 5.50201728e-02, 1.05100089e-01, 1.04517249e-01,
#   9.59753315e-02, 1.12985871e-01, 1.10426279e-01, 1.00069840e-01,
#   1.05789839e-01],
#  [ 4.82843013e-02, 9.79222686e-03,-1.62359353e-02,-1.41514371e-03,
#   9.96412256e-01, 2.13110073e-03, 8.23351635e-04, 1.65123612e-03,
#   5.41361200e-03,-1.10120535e-02, 1.11560919e-02, 6.14568903e-02,
#   6.67273585e-02, 7.05349713e-02, 1.16956251e-01, 1.14543643e-01,
#   1.05793274e-01, 1.18790455e-01, 1.19978346e-01, 1.12475627e-01,
#   1.17945931e-01],
#  [ 2.78111820e-02, 1.02881878e-02,-6.34095360e-03,-1.34821807e-02,
#   -1.01803729e-02, 9.91098912e-01,-9.91623063e-04, 1.86269418e-03,
#   -3.76721851e-03,-4.44984190e-03,-4.82902496e-04, 4.82842810e-02,
#   4.43074240e-02, 4.64605721e-02, 9.57168213e-02, 1.09676542e-01,
#   1.01201468e-01, 1.00752343e-01, 1.06245567e-01, 1.00547332e-01,
#   1.03062772e-01],
#  [ 1.68435134e-02, 8.74554308e-03,-1.39346188e-02,-1.10940754e-02,
#   -9.32544452e-03,-2.99800982e-04, 9.89154879e-01,-5.32850411e-03,
#   3.84422894e-03, 2.76520694e-03, 6.18921696e-03, 3.78757321e-02,
#   3.82088813e-02, 3.72337773e-02, 8.40310707e-02, 9.16393067e-02,
#   9.81521009e-02, 1.00622318e-01, 1.07006095e-01, 9.60115823e-02,
#   1.01455338e-01],
#  [ 2.40067238e-02, 4.45399320e-03,-8.17943813e-03,-7.00694929e-03,
#   -1.76924670e-03,-1.92623443e-03, 2.10152103e-03, 9.92977702e-01,
#   1.14850680e-03,-5.65898171e-03, 8.61855606e-03, 4.42926288e-02,
#   5.05211120e-02, 4.77261866e-02, 8.86295647e-02, 9.77450705e-02,
#   9.79092814e-02, 1.11030806e-01, 1.00907242e-01, 1.00731715e-01,
#   1.00048965e-01],
#  [ 1.65836083e-02, 2.51233291e-03,-3.33006078e-03,-5.52148318e-03,
#   4.57899968e-03,-5.67542593e-03,-7.67010181e-03,-1.02494723e-02,
#   1.00385835e+00, 3.28825559e-03, 1.22669532e-02, 4.01098966e-02,
#   3.68492273e-02, 3.88462965e-02, 8.71975599e-02, 9.73437734e-02,
#   9.16506181e-02, 9.89761416e-02, 1.08627148e-01, 9.37317955e-02,
#   1.04060885e-01],
#  [ 9.41051675e-03, 1.40638341e-03,-1.25185245e-02,-1.69600844e-02,
#   -1.66540509e-04,-1.13054506e-02,-3.16014146e-03,-2.52028738e-03,
#   5.16692939e-03, 9.85737010e-01, 1.08764907e-02, 3.51511407e-02,
#   3.57462632e-02, 4.04998631e-02, 8.02783496e-02, 9.01276060e-02,
#   8.61492743e-02, 9.42692288e-02, 9.21916410e-02, 1.02453283e-01,
#   8.97679469e-02],
#  [-3.32196896e-03, 1.70325496e-02,-1.41300379e-02,-1.02379392e-02,
#   -3.76473163e-03,-3.92312144e-03,-7.73470493e-03,-3.03316314e-04,
#   -8.92442133e-04,-2.81636232e-03, 1.00163923e+00, 1.58716140e-02,
#   1.92888047e-02, 2.27184938e-02, 7.58318336e-02, 8.41903718e-02,
#   8.15260120e-02, 9.57874107e-02, 8.84535488e-02, 8.11922674e-02,
#   1.04585303e-01]])


        
        
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




