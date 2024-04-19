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
        self.transition = np.array([[ 3.87859421e-01, 1.00852524e+00, 1.22571455e-04,-2.06891992e-04,
  1.73087647e-03,-2.44269556e-04, 4.02860423e-04, 5.15855247e-04,
  -3.51228467e-04, 8.81468462e-04, 1.56361210e-03,-6.40080079e-01,
  -1.28594513e-02,-1.50109661e-02,-2.90162472e-02,-2.05679655e-02,
  -3.71921795e-02,-1.57983431e-02, 1.56844262e-02,-2.44654413e-03,
  -5.71021211e-02],
 [ 4.43429557e-01, 7.61725484e-05, 1.00809341e+00, 5.75082206e-04,
  3.49598312e-04, 2.07761954e-04, 1.38117992e-03,-1.27706452e-03,
  6.40217114e-04, 1.18031259e-03,-5.24857014e-04, 2.78046712e-02,
  -5.59278083e-01, 1.27977187e-02, 1.87574708e-02,-8.17394709e-03,
  -1.00848190e-02, 4.23068806e-03,-8.36931610e-03,-1.82767915e-02,
  4.00161697e-02],
 [ 4.43378592e-01, 8.58256126e-04,-1.98628950e-04, 1.00995263e+00,
  1.11158038e-03,-1.51272673e-04, 4.91825595e-04, 6.90396900e-04,
  -3.73838291e-04,-1.59255411e-03, 1.19335052e-03, 2.11552339e-02,
  4.83134579e-03,-5.82384739e-01,-6.26717041e-03,-1.23517468e-03,
  -6.42355867e-06, 2.21508030e-03,-1.84550907e-02,-6.16632371e-03,
  -3.07562563e-02],
 [ 4.38501632e-01, 1.38508163e-03,-1.48060926e-03, 1.34304780e-03,
  1.00822405e+00,-5.64959113e-04, 3.36697122e-04, 9.68781894e-04,
  -1.28977976e-03,-7.33852849e-05, 3.16270594e-04,-9.09355637e-03,
  3.35467103e-02, 3.13791640e-02,-6.12831017e-01, 2.34913821e-02,
  -6.69692665e-02, 3.24257171e-02, 4.14833231e-02,-1.13727320e-02,
  -1.59254284e-02],
 [ 4.36471763e-01, 3.05582736e-04,-1.91718848e-04, 6.05376372e-04,
  8.86680441e-05, 1.00939691e+00,-2.38149064e-04, 8.80970172e-05,
  -8.24167310e-04, 4.14006646e-04, 1.31366154e-04, 1.17372424e-01,
  3.12669229e-02,-6.27457955e-04, 1.59665589e-02,-5.89709433e-01,
  -9.31650828e-02,-3.52044074e-02,-1.32159080e-02,-4.58369369e-02,
  -1.86736284e-02],
 [ 4.61875712e-01,-2.56030226e-04,-1.59373336e-04,-2.07061442e-04,
  6.26001151e-04, 9.70039492e-04, 1.00990542e+00,-3.70273580e-04,
  5.70980516e-04, 9.52717905e-04, 4.73897785e-04,-1.55265640e-02,
  5.82634915e-03,-3.82350932e-02,-3.21997493e-02,-1.99003608e-02,
  -6.11211173e-01,-4.43160430e-02,-1.08578693e-05,-1.54544933e-02,
  -1.29550867e-02],
 [ 4.40044864e-01, 1.62291343e-03, 8.19613839e-04, 9.11613303e-04,
  -1.24747335e-03,-1.66779148e-03, 5.57837796e-04, 1.00861444e+00,
  6.46837597e-04,-2.05086381e-04,-5.91537612e-04,-2.57208046e-03,
  6.78470217e-03,-1.05268783e-02, 1.91204262e-02, 2.69886572e-03,
  -2.78275518e-02,-5.76288182e-01,-1.77004489e-03,-5.38844878e-04,
  -2.51065511e-03],
 [ 3.92927860e-01, 4.73651187e-05, 3.58722211e-04,-2.39170183e-03,
  -5.07548854e-04, 1.16796269e-03,-9.19376166e-04, 2.79164192e-03,
  1.00786283e+00,-6.79318480e-04, 6.69646385e-05, 2.93903693e-03,
  4.82332980e-03, 2.64624926e-02, 8.09331178e-03,-1.02342837e-02,
  6.40869470e-04,-2.01532635e-02,-5.90905643e-01,-9.79758521e-03,
  3.58757204e-04],
 [ 5.41523107e-01, 1.67123323e-04,-1.50179733e-03,-6.04259623e-04,
  -1.53686094e-03, 1.60241407e-03, 1.48829514e-03, 7.14622945e-04,
  -6.05128989e-04, 1.00982502e+00, 2.00952583e-03,-3.95901929e-02,
  -7.95565511e-03,-3.10321352e-02, 3.50430918e-03,-2.12512650e-02,
  -7.56104908e-02,-1.91080711e-02,-3.17464094e-02,-5.89576815e-01,
  -1.01853755e-01],
 [ 4.86562935e-01, 5.06471079e-04, 5.16553834e-05,-1.97746942e-03,
  4.76778401e-04, 1.06900790e-03, 1.11370193e-04, 1.84821498e-04,
  1.14244342e-03, 1.23963930e-03, 1.01063137e+00,-1.04139388e-01,
  -3.52309576e-02,-1.54009304e-02,-1.53126229e-02,-2.31264259e-02,
  -9.41606356e-02,-7.12551897e-03,-2.68809961e-03,-4.04005665e-03,
  -6.16452983e-01]])

#         self.transition = np.array([[ 5.31371499e-01, 1.00629199e+00, 1.06100208e-03,-1.33804532e-03,
#   1.07632705e-03, 4.38172001e-04, 3.58071513e-04, 1.33815287e-03,
#   -5.57410345e-05, 2.20506683e-03, 1.32663202e-03,-6.11917798e-01,
#   -6.12148839e-02,-2.26198760e-02,-3.28042668e-02,-2.75220136e-02,
#   -5.29694833e-02,-2.85904618e-02,-1.37755790e-02,-1.29591018e-02,
#   -4.84567665e-02],
#  [ 5.57099011e-01, 1.25633472e-03, 1.00502529e+00, 7.57215147e-04,
#   4.74918342e-04, 5.47708843e-04, 1.65750981e-03,-5.82043432e-05,
#   1.06705584e-03, 8.76348707e-04, 1.53161094e-03,-1.00264999e-02,
#   -5.27838957e-01, 1.47824109e-02, 7.87651596e-03,-2.45465634e-02,
#   2.02598274e-02,-9.78533410e-03, 3.80950934e-03,-1.54507628e-02,
#   3.30015081e-02],
#  [ 5.80460530e-01, 1.35117222e-03, 1.47437995e-03, 1.00613194e+00,
#   1.26245774e-03,-4.49941360e-04,-2.39813585e-04,-1.18343441e-03,
#   -1.01099477e-03,-7.16806955e-04, 9.11996623e-04, 3.23893831e-02,
#   -4.65192571e-02,-5.57354908e-01,-2.54546418e-02, 1.76868782e-02,
#   -3.90673152e-02, 9.00813133e-03,-2.22469233e-02,-1.70146560e-02,
#   -4.55762374e-02],
#  [ 5.24376443e-01,-7.65355087e-04, 5.49884934e-04, 4.60098766e-04,
#   1.00445128e+00,-1.56809937e-03, 4.84746649e-04, 8.87461310e-04,
#   -5.47392349e-04, 9.13613092e-04, 2.94595821e-03,-8.45591207e-03,
#   4.22161080e-02, 1.61182099e-03,-5.59941357e-01, 3.04211993e-02,
#   -8.47237500e-02, 4.53954200e-02,-8.51584479e-03,-4.87112125e-03,
#   4.22246526e-03],
#  [ 4.78027162e-01,-3.19498256e-04, 1.62217848e-04, 1.52343785e-04,
#   -1.33790104e-03, 1.00499712e+00, 5.28878869e-04,-4.64934177e-04,
#   8.09910104e-04, 3.77765244e-04, 1.56759788e-03, 8.67146976e-02,
#   2.00081258e-02, 1.13064475e-02, 9.82510573e-03,-5.66910950e-01,
#   -3.96231038e-02,-8.09794568e-03, 7.67676140e-04,-3.58418275e-02,
#   -4.53912160e-02],
#  [ 5.18917054e-01,-6.69496906e-04, 8.81236157e-04,-1.26117290e-04,
#   8.68622661e-04, 1.16337598e-03, 1.00740125e+00, 6.03624109e-04,
#   1.27424475e-03,-5.73599104e-04, 4.82379352e-04,-2.26297267e-02,
#   -3.94418970e-02,-3.78347679e-02,-3.21594987e-02,-2.88405077e-02,
#   -5.41850349e-01,-4.64270912e-02,-3.69247806e-02,-4.74129086e-03,
#   -1.34509678e-02],
#  [ 5.16355580e-01, 4.88219079e-04, 9.24191139e-04,-1.08428461e-03,
#   2.26668104e-04, 4.46091471e-03,-1.83854787e-03, 1.00628131e+00,
#   -8.73259117e-04,-5.04303293e-04, 1.39469024e-03,-1.72951827e-02,
#   8.73564004e-03,-1.18690255e-02,-1.21280781e-02,-2.19095013e-02,
#   -3.42831747e-02,-5.41018230e-01,-7.08954407e-03, 1.05597494e-02,
#   -5.67007659e-03],
#  [ 5.28109527e-01,-7.28419048e-04,-1.70605102e-03,-1.47661126e-04,
#   -4.91208550e-05, 7.74051127e-04,-1.15897588e-03, 1.75461454e-03,
#   1.00464240e+00,-1.57638328e-03, 5.57389309e-04, 2.48667915e-02,
#   -2.30865027e-02,-8.77497182e-03, 6.09471967e-03,-2.32528055e-02,
#   -3.16786091e-04,-2.78737463e-02,-5.49265697e-01,-1.71049943e-02,
#   -2.33825566e-02],
#  [ 4.40477557e-01, 1.01236585e-04,-1.48079678e-03, 1.50085905e-04,
#   2.05791359e-03, 9.79840133e-04, 7.11154852e-04, 5.68600007e-05,
#   7.88992651e-04, 1.00708103e+00,-7.91913634e-04,-5.07737775e-02,
#   -2.25680209e-02, 2.83544065e-03,-9.56695983e-03,-4.78339134e-03,
#   -3.72393278e-02,-7.25879477e-03,-5.33129123e-03,-5.30701615e-01,
#   -5.63884052e-02],
#  [ 5.16959134e-01, 1.32060623e-03, 6.38937662e-04, 1.12966448e-03,
#   1.90008734e-04, 6.51683234e-05,-1.90612821e-03,-3.61151140e-05,
#   -1.19930638e-03, 1.87516239e-05, 1.00787884e+00,-1.27470651e-01,
#   -2.60145864e-02,-2.04698470e-03,-2.16489384e-02,-1.53814207e-02,
#   -4.99156553e-02,-3.59964958e-02,-2.97465050e-02,-2.01578814e-02,
#   -5.52070761e-01]])
  
  

        
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




