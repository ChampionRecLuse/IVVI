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
#         self.transition = np.array([[-7.98722828e-03, 1.00640662e+00,-1.56888449e-02,-3.64064015e-03,
#   -1.14703319e-02,-1.11066525e-02, 9.61758456e-03,-5.04247286e-03,
#   -2.25780848e-03,-3.73970745e-06,-6.38642443e-05, 3.64128533e-02,
#   3.53813063e-02, 3.43654925e-02, 3.53427738e-02, 3.75027473e-02,
#   3.79390708e-02, 3.82267068e-02, 3.74736532e-02, 4.48978615e-02,
#   3.79667079e-02],
#  [ 2.93998402e-02, 8.77115906e-03, 9.86205716e-01, 3.09483057e-03,
#   -3.46085200e-03,-8.24203921e-03,-4.02724081e-03,-7.63841771e-03,
#   -5.11735878e-03, 1.52255601e-02, 4.23724029e-03, 5.55601716e-02,
#   6.91430996e-02, 6.19935313e-02, 7.02019880e-02, 7.28199624e-02,
#   7.07681797e-02, 6.57137007e-02, 5.88443836e-02, 6.71902801e-02,
#   6.18571640e-02],
#  [ 2.65882286e-02, 3.57772228e-03,-1.41747002e-02, 9.99543657e-01,
#   -1.27214224e-03,-4.71088991e-03, 8.33200015e-04,-2.98788991e-03,
#   -4.34750226e-03,-3.08190223e-03,-1.27758391e-02, 6.71653076e-02,
#   6.15250162e-02, 7.02223460e-02, 6.50643366e-02, 5.92555847e-02,
#   6.65561583e-02, 6.42945088e-02, 6.56649592e-02, 6.56177838e-02,
#   5.75933383e-02],
#  [ 2.33943872e-02, 5.37164122e-03,-4.82030059e-03, 6.94306047e-03,
#   9.92527381e-01,-5.80784369e-03, 1.00000919e-03,-5.94758634e-03,
#   5.25421901e-03, 1.74285793e-02,-1.15885357e-03, 5.88919145e-02,
#   6.81660268e-02, 6.74719214e-02, 6.83235443e-02, 5.51731649e-02,
#   5.26039365e-02, 6.48168779e-02, 6.63926509e-02, 5.79839997e-02,
#   5.82622298e-02],
#  [ 1.59027378e-02, 1.06437509e-02,-1.39724922e-02,-8.81637587e-04,
#   -1.14050999e-02, 9.87816557e-01,-3.63519515e-03,-1.21788167e-02,
#   -8.71877512e-04, 5.70353856e-03,-2.72205067e-04, 4.72412928e-02,
#   5.66947620e-02, 5.06562254e-02, 5.83244738e-02, 6.32123368e-02,
#   5.47762041e-02, 5.94407825e-02, 5.07593084e-02, 5.88142765e-02,
#   5.09557838e-02],
#  [-8.90033390e-04, 6.15232193e-03,-1.12246004e-02, 6.35533741e-03,
#   -9.22923342e-03,-4.24152847e-03, 9.92662477e-01,-6.69228429e-03,
#   -5.95949277e-04,-5.34045762e-03, 4.89788134e-03, 4.18078508e-02,
#   4.16264843e-02, 4.67180399e-02, 4.90564217e-02, 4.49487500e-02,
#   4.14984211e-02, 4.02390547e-02, 4.64504037e-02, 3.50840764e-02,
#   4.56529466e-02],
#  [ 2.17866022e-02, 8.03471968e-03,-1.55200184e-02, 1.09156844e-02,
#   -1.59827903e-03,-8.50028764e-03,-2.63433062e-03, 9.93255762e-01,
#   5.58355242e-03, 1.14430328e-02, 7.77213999e-04, 5.92397111e-02,
#   6.60555866e-02, 6.68306813e-02, 6.07611309e-02, 5.87842953e-02,
#   6.17838314e-02, 6.48953388e-02, 6.25136080e-02, 5.77191579e-02,
#   6.03342641e-02],
#  [ 2.17882716e-02, 8.14634840e-03,-2.44311182e-02, 9.88719237e-03,
#   -1.64568422e-02,-2.41970042e-03, 5.34826799e-03,-1.29138911e-02,
#   9.91299845e-01, 2.44882957e-03, 2.52015023e-03, 6.15919711e-02,
#   6.13686612e-02, 5.68435130e-02, 5.99608234e-02, 5.94513287e-02,
#   6.58516680e-02, 5.66635728e-02, 6.71435722e-02, 6.69270114e-02,
#   4.55778739e-02],
#  [-3.33068623e-03, 1.47059230e-02,-9.45473273e-03, 3.24285215e-03,
#   -1.16942054e-02,-4.66487014e-03,-4.71971908e-03,-1.37246645e-02,
#   5.61396164e-04, 1.02134778e+00, 8.88790002e-03, 3.96425940e-02,
#   4.27900474e-02, 4.33365370e-02, 3.94629533e-02, 3.42547973e-02,
#   4.11451706e-02, 3.95001709e-02, 4.13454255e-02, 4.75547093e-02,
#   2.96289023e-02],
#  [-1.91832636e-02, 9.21916096e-03,-1.65508569e-02, 1.09386188e-02,
#   -9.55255645e-03,-1.26552451e-02, 1.21964496e-02,-1.82469119e-02,
#   -7.80644242e-03, 1.09891223e-02, 9.97553052e-01, 2.77043442e-02,
#   2.69588774e-02, 2.91827083e-02, 2.74731326e-02, 2.78795752e-02,
#   2.91155479e-02, 3.00661071e-02, 3.41883063e-02, 3.26050523e-02,
#   2.35854743e-02]])

        self.transition = np.array([[-6.46171138e-03, 1.01481509e+00,-7.09322697e-03, 8.82182778e-03,
  1.01757722e-02, 1.66102383e-04, 8.87193301e-03,-8.42232237e-03,
  -2.35228815e-03, 7.57954881e-03, 7.65516684e-03, 3.77440097e-02,
  3.99043255e-02, 3.07560977e-02, 3.96048534e-02, 4.22009179e-02,
  4.12986902e-02, 3.72176333e-02, 3.34586552e-02, 3.47067795e-02,
  3.74186795e-02],
 [ 3.01628474e-02, 2.44354961e-02, 9.92236980e-01,-4.14972435e-03,
  3.90722165e-03,-8.66602942e-04, 1.14861044e-02, 2.68559982e-04,
  2.05092588e-03,-3.07150505e-03, 1.13711401e-02, 6.04136717e-02,
  7.17521576e-02, 5.74240042e-02, 6.55239088e-02, 7.51980031e-02,
  6.90501710e-02, 6.81332685e-02, 6.46078481e-02, 7.02975597e-02,
  6.46188143e-02],
 [ 5.27735772e-02, 2.43000439e-02,-9.16509941e-03, 9.93376303e-01,
  -7.86030137e-03,-8.34196906e-03, 1.00977119e-02,-4.59220505e-03,
  -7.68671095e-03, 3.67941961e-03, 1.89190530e-02, 8.82066020e-02,
  8.74879516e-02, 8.73788958e-02, 7.47400016e-02, 8.38516496e-02,
  8.27541157e-02, 8.71346133e-02, 8.38477025e-02, 7.94457306e-02,
  8.66717508e-02],
 [ 2.33614722e-02, 2.00769665e-02,-4.46562380e-03,-4.83609581e-03,
  9.84474447e-01, 6.05440971e-03,-6.31774486e-03,-1.05100985e-03,
  -4.61350599e-03, 8.34531707e-03, 1.09025145e-02, 6.05938094e-02,
  5.90384398e-02, 6.18188222e-02, 6.92077673e-02, 6.23813301e-02,
  5.42523710e-02, 6.44881588e-02, 6.11967593e-02, 5.91161349e-02,
  6.24217878e-02],
 [ 2.15844036e-02, 2.07699155e-02,-1.94012615e-02,-5.41373241e-03,
  1.08357458e-02, 9.99404950e-01, 6.27007374e-03, 3.40882368e-03,
  -6.01266438e-04, 7.55755242e-03, 1.71743679e-02, 7.07506564e-02,
  6.60286970e-02, 5.38550022e-02, 6.36277754e-02, 5.58657875e-02,
  5.86245909e-02, 5.75171305e-02, 6.05190240e-02, 6.50139816e-02,
  5.95755640e-02],
 [ 9.26946406e-03, 2.41122887e-02,-6.89393839e-03,-1.22154403e-02,
  -2.64508089e-03,-1.29996967e-03, 1.00777323e+00, 2.80335438e-04,
  -4.05514076e-03, 3.75974779e-03, 1.13689961e-02, 5.43454632e-02,
  5.36057508e-02, 4.87927922e-02, 4.88492471e-02, 4.75517680e-02,
  5.59686019e-02, 5.23179058e-02, 5.29141054e-02, 6.13375668e-02,
  4.81388510e-02],
 [ 1.65909654e-02, 9.76486159e-03,-1.52263704e-02,-8.40912198e-03,
  3.29968936e-03, 5.63676759e-03, 1.10705415e-02, 9.88511568e-01,
  7.88920265e-04, 4.84030918e-03, 1.20065019e-02, 6.12942316e-02,
  5.14649176e-02, 6.01953467e-02, 5.23215080e-02, 5.20147027e-02,
  6.05745226e-02, 5.68677555e-02, 5.82594104e-02, 6.55670381e-02,
  5.13437385e-02],
 [ 2.33059890e-02, 1.38562134e-02,-1.05954811e-02, 2.16400825e-02,
  5.28728006e-04,-9.92734323e-04, 6.44968207e-03,-1.28235745e-02,
  9.83651450e-01, 9.62169509e-03, 6.52809593e-03, 6.14290748e-02,
  6.41477394e-02, 5.29235895e-02, 5.96970990e-02, 6.62355208e-02,
  6.53478943e-02, 6.72013572e-02, 6.20171534e-02, 5.46542639e-02,
  6.25615774e-02],
 [ 1.48634668e-03, 8.43904115e-03,-7.62538870e-03, 1.71348274e-03,
  3.99888876e-03, 2.37494848e-03, 7.33597237e-03,-1.31882335e-02,
  -1.26420304e-02, 9.99060394e-01, 1.20530576e-02, 4.96715165e-02,
  4.67554599e-02, 4.19816203e-02, 5.01769657e-02, 3.89198235e-02,
  4.58453063e-02, 4.59499727e-02, 4.22900527e-02, 3.99763106e-02,
  5.08193526e-02],
 [-2.33610986e-04, 2.00855003e-02,-8.87564970e-03,-9.64891441e-03,
  -7.23449377e-03,-2.52340189e-03, 6.60394975e-06, 3.69240928e-03,
  -3.26186111e-03,-1.22602322e-02, 1.00313473e+00, 3.89456023e-02,
  4.52170065e-02, 4.01195724e-02, 4.27550257e-02, 4.33571586e-02,
  3.79653092e-02, 5.11604447e-02, 4.29671762e-02, 3.75342908e-02,
  5.25473497e-02]])
   
   

        
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




