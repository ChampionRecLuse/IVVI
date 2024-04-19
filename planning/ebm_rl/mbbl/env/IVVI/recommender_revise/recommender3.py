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
        self.transition = np.array([[-1.44422730e-02, 1.01054497e+00,-1.18004737e-02,-1.43120254e-02,
  -8.02424327e-03,-2.17463298e-03,-9.98553578e-04, 4.09942842e-03,
  3.66711797e-04, 1.27475667e-02, 8.43825215e-03, 6.69609627e-02,
  5.74819512e-02, 6.23687101e-02, 4.80840267e-02, 6.43709015e-02,
  5.86876436e-02, 5.90105797e-02, 5.46148643e-02, 5.85855896e-02,
  5.66821047e-02],
 [ 3.23559655e-02, 7.09115807e-03, 9.92485350e-01,-9.87819552e-03,
  -2.34068711e-03,-2.44892955e-03, 5.01679072e-03,-7.12159963e-03,
  2.23995993e-03, 1.45316294e-02,-2.23174206e-03, 8.77008813e-02,
  8.77937856e-02, 8.97462215e-02, 8.81466098e-02, 8.42799110e-02,
  7.99773347e-02, 8.61171916e-02, 8.35035216e-02, 8.42750214e-02,
  9.37572033e-02],
 [ 3.59832734e-02, 1.56028878e-02,-1.33749657e-02, 9.91397495e-01,
  -5.08836606e-03,-6.42209762e-03,-1.35534186e-03, 1.06999156e-03,
  -4.55299421e-03, 6.17533377e-03, 1.06393715e-02, 8.39749312e-02,
  8.90752952e-02, 9.81373507e-02, 8.82959699e-02, 9.07219088e-02,
  8.78881876e-02, 9.17253694e-02, 9.31880455e-02, 9.01446862e-02,
  8.64701439e-02],
 [ 2.43498907e-02, 6.98026755e-03,-3.51925827e-03,-4.35336672e-03,
  9.92096650e-01,-9.88148575e-04, 2.70278191e-03, 5.71500391e-03,
  -7.68948042e-04, 1.58786275e-02, 4.09023118e-03, 8.65299932e-02,
  8.47475241e-02, 7.99039569e-02, 8.37631062e-02, 7.75687853e-02,
  7.57374210e-02, 8.92030862e-02, 7.90899328e-02, 8.12516518e-02,
  8.08636021e-02],
 [ 2.58313800e-02, 1.27765228e-02,-1.08758187e-02,-1.01490111e-02,
  -1.20673694e-02, 9.99375667e-01,-6.20976171e-04,-5.76526896e-03,
  -3.26758314e-03, 7.67714140e-03, 1.15105534e-02, 8.52621718e-02,
  7.93334639e-02, 8.39535025e-02, 7.91395728e-02, 9.02438283e-02,
  8.04154811e-02, 8.26269929e-02, 8.12073958e-02, 8.09721650e-02,
  8.37915050e-02],
 [ 7.83797039e-03, 9.79895351e-03, 1.87028307e-04,-7.70729774e-03,
  -5.49372710e-03, 4.45261604e-03, 1.00039463e+00, 5.42266643e-04,
  1.87432544e-03, 1.37394386e-02, 4.82315400e-03, 7.42533071e-02,
  6.85803652e-02, 6.77203247e-02, 7.12915114e-02, 6.82747103e-02,
  7.99540345e-02, 6.84824975e-02, 6.88221953e-02, 6.51619054e-02,
  8.02397858e-02],
 [ 2.43246880e-02, 6.96210015e-03, 2.39530162e-03,-1.28314048e-02,
  -9.09755397e-03,-6.65244927e-03,-1.13136368e-03, 1.00194673e+00,
  -8.38634173e-03, 3.26636291e-03,-2.31667945e-04, 8.04915491e-02,
  7.45540336e-02, 8.34261193e-02, 7.90468846e-02, 7.85671910e-02,
  8.09567029e-02, 8.99990218e-02, 8.02271045e-02, 8.53283943e-02,
  7.91665974e-02],
 [ 3.37624103e-02,-2.02643608e-03,-5.37424714e-04,-1.64242093e-02,
  -5.26814505e-03, 4.97030619e-04, 9.03585367e-04, 9.09325084e-03,
  1.00152793e+00, 1.14281113e-02, 3.28948111e-03, 8.44604817e-02,
  8.73340187e-02, 8.78054316e-02, 8.51480132e-02, 9.11502853e-02,
  9.31639039e-02, 8.73127778e-02, 9.48461902e-02, 8.31755171e-02,
  8.68023886e-02],
 [ 1.29285955e-02, 3.00207032e-03,-4.43466372e-03,-2.05543756e-02,
  -8.01237185e-03, 2.90276945e-05,-8.91520445e-04, 2.50386158e-03,
  -4.33124597e-03, 1.00804759e+00,-7.86296977e-04, 7.43596259e-02,
  7.52049341e-02, 8.13574936e-02, 7.43962495e-02, 7.43927155e-02,
  7.56254216e-02, 6.93893529e-02, 7.21180201e-02, 7.50479981e-02,
  7.22170225e-02],
 [-8.08016882e-03, 3.02555717e-03,-4.98778542e-03,-1.74009055e-02,
  -2.62696718e-03,-2.14624349e-03,-3.04179342e-03,-1.98088613e-03,
  2.03275424e-03,-1.79929091e-03, 1.00020016e+00, 5.95117264e-02,
  6.22120164e-02, 6.17549410e-02, 5.99028557e-02, 5.64338618e-02,
  5.95975392e-02, 5.71952517e-02, 5.91802163e-02, 6.03414115e-02,
  6.91065355e-02]])

#         self.transition = np.array([[ 1.29636873e-03, 9.89792362e-01,-1.95874958e-02,-1.24031456e-02,
#   -3.92757400e-03, 5.33652810e-03, 3.08528599e-03,-1.43626864e-03,
#   -9.88974951e-05, 1.91723533e-02, 1.02865162e-02, 4.79207372e-02,
#   4.80654162e-02, 4.30366145e-02, 4.40786409e-02, 4.63594314e-02,
#   3.62168539e-02, 4.98086272e-02, 4.69689005e-02, 4.39395595e-02,
#   4.43102042e-02],
#  [ 4.82190165e-02, 3.28708222e-03, 9.73259605e-01,-1.76751121e-03,
#   -1.17677610e-03, 4.21401935e-03, 6.80391441e-03, 1.14997556e-02,
#   -5.61713212e-03, 1.91931720e-02, 2.94162074e-03, 7.77857137e-02,
#   8.55482501e-02, 7.89045927e-02, 8.41102609e-02, 8.32184063e-02,
#   7.76376712e-02, 7.74377268e-02, 8.51115264e-02, 7.76092621e-02,
#   7.99677745e-02],
#  [ 3.68877680e-02, 3.14838310e-04,-1.90092927e-02, 9.96008650e-01,
#   -5.96461378e-03, 1.52429180e-02, 4.32111109e-03, 8.79604427e-03,
#   3.82956695e-03, 1.80990582e-02, 3.08850411e-03, 6.87284720e-02,
#   6.94468141e-02, 7.80484617e-02, 7.42269026e-02, 7.03904541e-02,
#   7.27057010e-02, 7.87122467e-02, 7.09369736e-02, 6.69165120e-02,
#   7.29166280e-02],
#  [ 3.77566477e-02,-4.91336743e-03,-1.11386218e-02,-4.80582867e-03,
#   9.95799100e-01, 6.57395174e-03, 5.80204513e-03, 5.50640445e-03,
#   5.49004419e-03, 1.65890609e-02, 5.15330540e-03, 6.45527465e-02,
#   7.41868862e-02, 7.41471952e-02, 8.19704700e-02, 7.58159135e-02,
#   6.96505403e-02, 7.64753075e-02, 7.29617774e-02, 6.58070845e-02,
#   7.40319419e-02],
#  [ 3.72202801e-02, 1.74020777e-02,-1.43599638e-02, 2.71105974e-04,
#   -1.45003248e-02, 9.99653457e-01, 7.31298175e-03, 4.35461403e-03,
#   4.31113827e-03, 2.53636771e-02, 1.01181260e-02, 7.95470566e-02,
#   7.44600090e-02, 7.10788637e-02, 7.09333257e-02, 7.68192056e-02,
#   7.03330610e-02, 7.25878999e-02, 7.58033103e-02, 6.87080340e-02,
#   7.01104003e-02],
#  [ 7.85742572e-03, 1.33617538e-03,-2.86140656e-02, 5.53543184e-03,
#   1.06137556e-02, 1.58251686e-02, 9.96952986e-01, 5.21100966e-03,
#   6.09842004e-03, 1.39064667e-02, 1.04523591e-02, 5.10938085e-02,
#   4.83766824e-02, 4.50456725e-02, 5.51807313e-02, 5.11797793e-02,
#   5.76932752e-02, 5.18448744e-02, 4.54391693e-02, 4.94044187e-02,
#   5.19603865e-02],
#  [ 4.08807081e-02,-4.30969227e-03,-1.74452147e-02,-6.45254960e-03,
#   6.19326372e-03, 1.50680196e-02,-3.50856283e-04, 1.00866908e+00,
#   -8.58952079e-03, 1.47762092e-02, 4.87710578e-03, 7.34134517e-02,
#   6.85541821e-02, 7.12450872e-02, 7.78395357e-02, 7.96049849e-02,
#   6.83325985e-02, 7.73456587e-02, 7.48751583e-02, 7.63091961e-02,
#   7.15201052e-02],
#  [ 3.71957832e-02,-1.27824624e-03,-1.28215275e-02, 1.38640274e-03,
#   -2.52242802e-03, 3.70766815e-03, 1.21302065e-02, 1.18537245e-02,
#   9.83915227e-01, 1.81986387e-02, 6.76809083e-03, 6.95745023e-02,
#   7.34452225e-02, 7.05726933e-02, 7.32920587e-02, 7.36985930e-02,
#   7.23165243e-02, 7.23863072e-02, 8.30272275e-02, 7.09566495e-02,
#   7.20336468e-02],
#  [ 1.28378664e-02,-1.55224837e-02,-1.75179110e-02,-1.10864642e-03,
#   -1.20271194e-02, 3.28307836e-03, 1.24293656e-02, 4.33600865e-03,
#   5.04324251e-03, 1.01992528e+00, 3.98119674e-03, 5.44043346e-02,
#   5.59467478e-02, 5.45587081e-02, 5.51335481e-02, 5.35178059e-02,
#   5.06618408e-02, 5.67600337e-02, 5.68950251e-02, 5.23759280e-02,
#   5.38562189e-02],
#  [ 5.18063694e-03, 7.79875308e-04,-1.20947586e-02, 7.63478245e-03,
#   -6.37801067e-03, 4.36011502e-03,-3.75315441e-03, 6.33484988e-03,
#   -6.59157378e-03, 1.36114385e-02, 1.01199148e+00, 4.76096115e-02,
#   4.90215950e-02, 4.38328554e-02, 4.95054362e-02, 4.64203971e-02,
#   4.56850653e-02, 5.43972545e-02, 5.03923050e-02, 4.09938877e-02,
#   5.15093900e-02]])




        
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




