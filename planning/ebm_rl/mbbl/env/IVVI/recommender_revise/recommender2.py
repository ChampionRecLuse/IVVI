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
#         self.transition = np.array([[-1.92625593e-03, 1.00095951e+00, 2.58623333e-03,-3.05416296e-03,
#   -2.63861741e-02, 3.56829895e-03, 1.18111898e-02,-4.68117752e-03,
#   -1.13918333e-02, 1.13275915e-02,-2.25188615e-03, 4.16922797e-02,
#   4.55906745e-02, 6.13312477e-02, 6.37839908e-02, 6.48468449e-02,
#   6.64512804e-02, 7.09161776e-02, 7.81214858e-02, 8.20021540e-02,
#   7.72887117e-02],
#  [ 3.98209522e-02, 1.79734964e-02, 1.00776099e+00, 3.98801924e-03,
#   -2.04927051e-02, 5.81479182e-03, 1.45866601e-02, 6.48714927e-03,
#   -7.75996316e-03, 7.90485805e-03, 5.92517039e-04, 7.99214780e-02,
#   8.13165346e-02, 9.75469759e-02, 9.46035118e-02, 9.10475313e-02,
#   9.79909793e-02, 9.42857218e-02, 1.01016991e-01, 9.22951717e-02,
#   9.65167559e-02],
#  [ 3.34008117e-02, 1.73330378e-02,-1.50915511e-03, 9.97074382e-01,
#   -1.56385786e-02, 1.87170884e-02, 9.34988784e-03, 3.04229720e-03,
#   -9.02216671e-03, 7.86979856e-03,-5.63685712e-03, 6.95136145e-02,
#   7.30332458e-02, 9.59557982e-02, 8.57525943e-02, 9.09712770e-02,
#   8.62358613e-02, 9.32731137e-02, 9.06300457e-02, 9.12301902e-02,
#   9.29031707e-02],
#  [ 3.31225912e-02, 7.86004575e-03, 1.94547416e-03, 1.29703729e-03,
#   9.81565654e-01, 2.14119098e-02, 1.53850773e-02, 4.72044847e-03,
#   -1.32603913e-02, 9.85412945e-03, 9.92513938e-03, 6.62524764e-02,
#   7.08022876e-02, 9.23443725e-02, 9.54368677e-02, 9.76198241e-02,
#   9.04588186e-02, 9.20744470e-02, 8.99718665e-02, 9.63267898e-02,
#   9.71799810e-02],
#  [ 4.40112845e-02, 9.66603790e-03, 2.07330020e-03, 2.19874202e-03,
#   -1.54326694e-02, 1.01117281e+00, 2.00396400e-02, 3.19001086e-04,
#   -2.15949197e-02, 1.10705381e-02,-1.79570874e-03, 8.27541124e-02,
#   8.30161905e-02, 9.21215266e-02, 9.56607268e-02, 1.02336989e-01,
#   8.99261875e-02, 1.06194500e-01, 9.74403705e-02, 1.02468172e-01,
#   9.98397884e-02],
#  [ 6.22769664e-03, 2.13887855e-03, 3.35858878e-03,-5.83619267e-03,
#   -6.87575639e-03, 6.10690958e-03, 1.01886542e+00, 2.12810740e-03,
#   -1.10091078e-02, 9.91286221e-03, 6.52724264e-03, 4.62880849e-02,
#   5.32324898e-02, 6.23495739e-02, 7.05153244e-02, 7.48183835e-02,
#   7.92460591e-02, 7.15875378e-02, 7.96673489e-02, 7.77840151e-02,
#   8.18741913e-02],
#  [ 3.98184167e-02, 2.26485471e-02, 4.08894036e-04,-6.02187162e-03,
#   -2.84166141e-02, 1.16243447e-02, 1.07736415e-02, 9.98562882e-01,
#   -2.23398907e-02, 2.46356396e-02, 1.12207049e-02, 7.12673055e-02,
#   7.98645669e-02, 8.57755291e-02, 9.46461281e-02, 8.59628961e-02,
#   8.98701207e-02, 1.08541603e-01, 9.45188175e-02, 9.87031785e-02,
#   9.70693329e-02],
#  [ 2.70693131e-02, 9.48230106e-03,-3.54773777e-03,-3.82607744e-03,
#   -7.86867996e-03, 1.05639162e-02, 1.56855484e-02, 2.55018197e-03,
#   9.80902459e-01, 1.51281520e-02, 6.30396045e-03, 6.97527396e-02,
#   6.53080089e-02, 8.15874089e-02, 8.26753044e-02, 8.48091515e-02,
#   8.34508296e-02, 9.09830178e-02, 9.89179860e-02, 8.92856188e-02,
#   9.25005390e-02],
#  [-8.22264819e-04, 5.19372816e-03,-1.20041027e-03,-5.48356378e-03,
#   -1.43880108e-02, 1.15317372e-02, 1.37369668e-02,-5.43310278e-04,
#   -6.99347531e-03, 1.01070580e+00,-1.96128978e-03, 5.17118340e-02,
#   5.18240805e-02, 6.11975765e-02, 6.51671119e-02, 6.56011570e-02,
#   6.57655969e-02, 7.59977437e-02, 7.13961802e-02, 8.77724576e-02,
#   8.15212018e-02],
#  [ 3.61835526e-03, 6.47620532e-03,-2.31094163e-03, 4.67700260e-03,
#   -1.02726294e-02, 7.84493813e-03, 1.75005976e-02, 2.31462874e-03,
#   -1.38986731e-02, 1.59494423e-02, 1.00949387e+00, 4.66465751e-02,
#   5.63335050e-02, 6.16722538e-02, 6.75448861e-02, 6.12772282e-02,
#   6.64357255e-02, 7.40256705e-02, 7.94491406e-02, 7.76903935e-02,
#   8.44770803e-02]])
   
        self.transition = np.array([[-2.23919018e-03, 1.00370963e+00,-1.06140700e-03,-4.76120226e-04,
  -2.81938491e-03,-1.44025328e-02, 6.49328866e-03,-3.95513412e-03,
  2.35588665e-03,-8.22682765e-03, 6.67722140e-03, 1.53325670e-02,
  2.15922647e-02, 4.14574257e-02, 4.22538153e-02, 4.72685577e-02,
  4.74134319e-02, 5.99491206e-02, 6.76510659e-02, 6.31869440e-02,
  6.52983052e-02],
 [ 4.94460222e-02, 1.06117275e-02, 9.87727673e-01,-5.39563089e-03,
  -1.17096990e-02,-6.89522311e-04,-1.96821023e-03,-7.44245273e-03,
  -4.70779950e-03, 3.39832456e-03, 1.01751740e-02, 6.40928465e-02,
  7.39143472e-02, 7.50347735e-02, 8.59918078e-02, 8.27386123e-02,
  9.10931856e-02, 8.76057509e-02, 9.53593045e-02, 8.56881884e-02,
  1.00479092e-01],
 [ 2.27212586e-02, 2.34141423e-03,-3.69371666e-03, 9.91129687e-01,
  -1.10745596e-02, 3.76798752e-04, 7.91495292e-03,-1.24062909e-03,
  -6.02850845e-06, 8.34258261e-03, 9.47634052e-03, 4.30099705e-02,
  4.26269455e-02, 6.72785186e-02, 6.92738969e-02, 6.07408848e-02,
  6.25527362e-02, 6.69600504e-02, 7.58445474e-02, 7.54166731e-02,
  7.26536078e-02],
 [ 4.27349617e-02, 3.48352005e-04, 9.85028332e-04,-9.73199977e-03,
  9.92271499e-01,-8.49353826e-03,-1.24494215e-02,-6.57599043e-03,
  6.40220904e-03, 6.62308443e-03,-3.53557531e-03, 6.08107163e-02,
  5.97199497e-02, 7.24731683e-02, 8.62991408e-02, 7.67865112e-02,
  7.25405453e-02, 8.91575785e-02, 9.57864721e-02, 8.53612130e-02,
  8.72313808e-02],
 [ 4.18069375e-02,-1.34863274e-02,-4.30694913e-03,-3.99437984e-03,
  -3.44096622e-03, 1.00114142e+00, 1.68808812e-03,-1.21584148e-02,
  -1.61304772e-03, 6.29525737e-03, 1.03774288e-02, 6.14265763e-02,
  6.05054892e-02, 7.25279377e-02, 7.78170396e-02, 8.72835749e-02,
  7.45341962e-02, 8.92317491e-02, 9.03102527e-02, 8.52324259e-02,
  8.37173919e-02],
 [ 2.30774790e-02, 3.86139278e-03, 7.03869983e-05,-9.84033501e-03,
  -6.12752076e-03, 9.13206151e-04, 9.97300512e-01,-3.13173488e-03,
  -3.40155642e-03,-8.23710963e-03, 6.56571929e-03, 4.01097816e-02,
  4.86673591e-02, 5.97210392e-02, 6.82824677e-02, 6.80566899e-02,
  6.96773784e-02, 6.78384890e-02, 8.08494553e-02, 7.25096077e-02,
  8.86937270e-02],
 [ 4.13088650e-02, 5.03217917e-03,-4.79179825e-03,-3.22854592e-04,
  1.89258738e-04,-8.78905214e-03, 3.38290579e-04, 9.87297844e-01,
  6.16001285e-04, 9.43494067e-04, 9.21087942e-03, 5.79154622e-02,
  6.44518650e-02, 7.56714877e-02, 7.96506587e-02, 7.67796847e-02,
  7.45489812e-02, 9.10372360e-02, 9.15032469e-02, 8.97692864e-02,
  8.80612301e-02],
 [ 3.28669241e-02, 6.35332211e-03, 5.36034902e-03, 1.07807608e-03,
  -8.75255300e-03,-7.64765119e-03, 1.48356466e-04,-1.00349374e-02,
  9.90693036e-01, 5.11891891e-03, 1.17702731e-02, 5.35263921e-02,
  5.15216670e-02, 6.92537648e-02, 7.38564195e-02, 7.46765211e-02,
  6.98596439e-02, 7.99173347e-02, 9.27658144e-02, 8.57123841e-02,
  8.36916006e-02],
 [ 1.41672441e-02,-2.29968679e-03,-1.67960317e-03,-4.53694400e-03,
  -1.63575439e-02, 4.90375158e-03,-7.98354210e-04,-1.13421321e-02,
  -1.03072061e-03, 1.00098987e+00, 1.11407414e-02, 3.06460254e-02,
  3.88306598e-02, 5.20078720e-02, 6.01218977e-02, 6.02373439e-02,
  6.44078431e-02, 7.35842544e-02, 7.57894632e-02, 7.64362274e-02,
  7.49406431e-02],
 [-8.69481441e-03, 4.97044876e-03,-1.04319093e-04, 1.50635087e-03,
  -3.03884292e-03,-4.39980800e-03,-4.00321490e-03,-2.85181701e-03,
  -1.40783872e-03, 6.00563568e-03, 9.97051417e-01, 1.21266900e-02,
  1.63263674e-02, 4.04370739e-02, 4.60935938e-02, 3.46893012e-02,
  3.80989479e-02, 5.57972924e-02, 6.06262796e-02, 5.86122556e-02,
  6.42806791e-02]])

        

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




