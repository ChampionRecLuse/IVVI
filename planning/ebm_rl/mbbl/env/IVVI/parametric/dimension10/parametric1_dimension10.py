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
#         self.transition = np.array([[ 2.14028032e-04, 5.05463646e-01, 2.04112072e-01, 2.99065024e-03,
#   -3.23288539e-03, 8.11640262e-03,-1.10136663e-02,-1.54313925e-02,
#   1.40691957e-03,-4.27392031e-04, 1.59212708e-02,-2.99474939e-01,
#   -2.60971383e-02, 2.64453021e-02, 2.43434028e-02, 3.72525860e-02,
#   2.06777824e-02, 3.53210359e-02, 3.57192435e-02, 2.96901937e-02,
#   2.97073259e-02],
#  [ 5.68004168e-03, 1.99795493e-01, 4.93358592e-01, 2.09764669e-01,
#   -7.79924413e-03,-3.43967071e-03,-1.45070682e-03,-8.60795605e-03,
#   1.57556790e-03,-4.24313743e-03,-6.85525745e-03,-2.17468715e-02,
#   -2.95318321e-01,-4.53313065e-02, 2.54499649e-02, 3.58210357e-02,
#   2.39721820e-02, 3.54797986e-02, 3.67265389e-02, 2.81349770e-02,
#   3.54875688e-02],
#  [ 1.61801100e-03,-6.72185316e-03, 2.14148420e-01, 5.13728099e-01,
#   1.92439641e-01,-1.56586133e-04, 6.98998632e-03,-6.52601182e-03,
#   -7.24514704e-03, 4.02597546e-03, 3.58379220e-04, 4.12030975e-02,
#   -3.79060684e-02,-3.05441660e-01,-3.04236168e-02, 2.95824894e-02,
#   3.04999503e-02, 3.43888403e-02, 3.26026744e-02, 3.91998173e-02,
#   3.89460601e-02],
#  [ 6.12600536e-04,-2.96568033e-03,-1.35403875e-04, 1.96344534e-01,
#   5.06205243e-01, 1.96453104e-01,-1.38163829e-02, 3.91555160e-03,
#   -4.42245870e-03, 2.12684784e-03, 6.60647399e-03, 3.93335515e-02,
#   4.06959748e-02,-1.79959978e-02,-3.07366682e-01,-2.09321450e-02,
#   2.81690656e-02, 3.81818095e-02, 3.46425882e-02, 3.86037894e-02,
#   4.45044385e-02],
#  [ 2.63587271e-03, 1.77031783e-03,-1.06055205e-03,-6.41456690e-03,
#   2.14913729e-01, 5.06678703e-01, 1.91366133e-01, 2.92995989e-03,
#   -2.50657173e-03,-6.22721362e-03, 4.56494424e-03, 3.73411961e-02,
#   4.03549996e-02, 3.10952695e-02,-3.18325061e-02,-3.07272805e-01,
#   -3.38559848e-02, 2.73069192e-02, 2.41063948e-02, 4.15734031e-02,
#   4.27294485e-02],
#  [-1.41622380e-03, 1.61812424e-03, 6.56359520e-04,-3.99571814e-03,
#   3.57018112e-03, 2.11979614e-01, 4.97957685e-01, 2.12336058e-01,
#   8.37337969e-03,-6.49301153e-03, 1.09749786e-03, 4.56977871e-02,
#   2.98972007e-02, 3.62292389e-02, 2.91170913e-02,-3.48799918e-02,
#   -2.94185905e-01,-3.17341874e-02, 2.52209690e-02, 3.76489369e-02,
#   4.12297194e-02],
#  [-1.01557525e-04,-1.22933509e-02,-3.75251757e-03, 3.93693239e-03,
#   3.54278712e-03, 3.42689448e-03, 1.85489590e-01, 5.10272672e-01,
#   1.95732687e-01, 7.37812351e-03,-2.98094253e-03, 2.04072237e-02,
#   3.39708248e-02, 3.40816932e-02, 2.42436864e-02, 3.87962033e-02,
#   -2.89776422e-02,-2.89466789e-01,-4.02687964e-02, 4.44195720e-02,
#   4.21828604e-02],
#  [ 3.37126488e-03,-1.43281762e-02, 4.25356866e-03, 1.59244558e-03,
#   9.19476840e-03, 7.66490738e-03, 5.99589974e-03, 1.97854748e-01,
#   4.87103373e-01, 1.98725880e-01, 7.10357982e-03, 2.63307101e-02,
#   3.04669370e-02, 2.80906314e-02, 3.27541655e-02, 4.52026849e-02,
#   2.90210663e-02,-1.11786489e-02,-2.97319535e-01,-2.54451447e-02,
#   3.77007712e-02],
#  [ 3.05407325e-03,-2.51301144e-03,-2.84825511e-03,-1.12186176e-02,
#   6.92613469e-03,-8.45056370e-03, 1.48329502e-03,-1.09047559e-03,
#   1.96555619e-01, 4.97484451e-01, 2.09809986e-01, 3.07161827e-02,
#   1.85942754e-02, 2.93850679e-02, 3.42047690e-02, 2.77540358e-02,
#   3.61211092e-02, 4.13438467e-02,-3.54345879e-02,-2.88098802e-01,
#   -2.81327034e-02],
#  [-1.54034499e-03,-4.04344101e-03,-8.53220154e-03,-1.15418111e-02,
#   -2.03744206e-03, 3.18110705e-03,-2.57308731e-03,-7.90507203e-03,
#   -2.80487199e-03, 2.08743123e-01, 5.10918927e-01, 3.60245781e-02,
#   2.83255656e-02, 2.44560498e-02, 3.49762676e-02, 3.99424609e-02,
#   3.27300207e-02, 4.41487921e-02, 3.44836214e-02,-2.91683972e-02,
#   -2.86962718e-01]])
  
  
        self.transition = np.array([[ 8.44388683e-03, 4.97553560e-01, 1.95478232e-01,-6.51571041e-03,
  -6.43311357e-03, 4.88968184e-03, 1.11197585e-03,-5.99476295e-03,
  -1.33295617e-03, 2.87242747e-03,-1.18658378e-02,-2.95579465e-01,
  -2.13597125e-02, 2.74645649e-02, 3.88258191e-02, 2.12825881e-02,
   2.62020662e-02, 3.36451858e-02, 3.65566088e-02, 2.37278365e-02,
   4.41766086e-02],
 [ 7.02322377e-03, 1.98797768e-01, 4.96828110e-01, 1.98459127e-01,
   4.34147497e-03,-4.21523755e-03, 6.92814841e-03,-3.96667707e-03,
  -1.06188512e-02, 4.12391583e-03,-1.26089962e-02,-2.79659076e-02,
  -2.94042376e-01,-2.81323403e-02, 1.89225337e-02, 3.31429689e-02,
   4.34627909e-02, 3.70272767e-02, 2.83983604e-02, 2.71448936e-02,
   4.05090492e-02],
 [ 8.41488197e-03, 1.04659449e-02, 2.05435140e-01, 5.03457574e-01,
   1.93276874e-01, 1.11004310e-02,-3.98277495e-03, 9.53407098e-05,
   6.08901868e-04,-9.01467340e-03, 1.94269067e-03, 4.28533218e-02,
  -2.31024054e-02,-3.03545432e-01,-4.08733847e-02, 3.29722936e-02,
   3.93219797e-02, 3.60915001e-02, 5.10019740e-02, 3.45916838e-02,
   3.09836184e-02],
 [ 8.80627867e-04, 1.33292994e-02, 2.67525215e-03, 2.08898869e-01,
   5.00061051e-01, 2.05331331e-01, 4.39597940e-03, 1.50410804e-02,
   5.71150213e-03, 3.17057406e-03, 8.34534206e-04, 3.60041797e-02,
   3.85821626e-02,-2.52966653e-02,-2.94382224e-01,-3.03972802e-02,
   3.09442020e-02, 4.31510408e-02, 3.89988106e-02, 4.19195750e-02,
   3.10977404e-02],
 [-3.84368476e-03, 1.44472836e-03,-8.29980572e-03, 8.91916189e-03,
   2.02815073e-01, 5.21523877e-01, 1.91988099e-01,-3.10663693e-03,
   1.10527698e-02,-1.35237211e-03,-7.22884097e-03, 3.55271777e-02,
   4.36277538e-02, 3.71252650e-02,-3.04996476e-02,-2.93559058e-01,
  -2.91900993e-02, 4.53876167e-02, 4.53343118e-02, 3.46942358e-02,
   4.46209616e-02],
 [ 9.90491604e-03,-1.58358195e-02,-2.64516640e-03,-2.62581194e-03,
  -6.09401267e-03, 2.14739257e-01, 4.96162085e-01, 2.03005542e-01,
  -3.77462157e-03, 1.46992949e-04,-6.78276790e-03, 3.57361652e-02,
   3.60494603e-02, 3.17753176e-02, 3.22986143e-02,-3.40420054e-02,
  -2.96940914e-01,-1.74213890e-02, 3.03286760e-02, 2.53716993e-02,
   3.93032520e-02],
 [ 1.55430106e-03,-1.25375921e-02,-1.17769094e-02, 9.41388977e-03,
  -3.99466777e-03,-2.12180352e-03, 1.90719625e-01, 4.92256479e-01,
   1.87849062e-01,-5.33416456e-03, 4.39062064e-03, 4.55994720e-02,
   3.61189064e-02, 3.36450136e-02, 2.75814588e-02, 1.37851664e-02,
  -3.07383734e-02,-3.03210871e-01,-3.09947123e-02, 3.49971290e-02,
   3.52543954e-02],
 [ 7.44727390e-03, 9.08460058e-03, 4.51521981e-03, 1.01487163e-02,
   1.09606073e-04,-3.28631165e-03,-1.65747129e-02, 2.01370653e-01,
   4.99612687e-01, 1.99989400e-01, 5.97280296e-03, 3.49854380e-02,
   4.85822881e-02, 3.21169160e-02, 1.74017263e-02, 3.03164577e-02,
   3.44443192e-02,-3.88077214e-02,-3.01824167e-01,-3.32665484e-02,
   1.94458677e-02],
 [ 2.80383563e-03, 1.66867700e-03,-6.43565887e-03,-2.16126668e-03,
   1.16757618e-03,-5.13577647e-03,-1.70303925e-02, 1.36267571e-02,
   2.06892923e-01, 4.94636361e-01, 2.10253696e-01, 4.64600407e-02,
   4.05158502e-02, 2.53751940e-02, 1.67331366e-02, 2.47225316e-02,
   3.79437503e-02, 2.71026973e-02,-3.43787390e-02,-3.03013583e-01,
  -2.88849715e-02],
 [ 3.86986694e-03,-3.97451758e-03,-3.37896301e-03, 9.83320471e-03,
   5.03092009e-03,-1.46591599e-02,-2.09261071e-03,-8.07909273e-04,
  -2.35861429e-03, 1.98727640e-01, 4.97409591e-01, 4.33469699e-02,
   3.29555162e-02, 3.18129629e-02, 3.77673367e-02, 3.60083973e-02,
   3.69502953e-02, 3.64994123e-02, 2.51297569e-02,-3.35354631e-02,
  -3.07427255e-01]])
    

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




