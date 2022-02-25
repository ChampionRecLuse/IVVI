from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import re

import numpy as np
import tensorflow.compat.v1 as tf
from brac import utils
from brac import agents
from brac import train_eval_ebm
from brac import train_eval_utils

env_name = 'IVVI'
# Create tf_env to get specs.
tf_env = train_eval_utils.env_factory(env_name)
tf_env_test = train_eval_utils.env_factory(env_name)
observation_spec = tf_env.observation_spec()
action_spec = tf_env.action_spec()

total_train_steps = int(1e6),
summary_freq = 100,
print_freq = 1000,
save_freq = int(1e8),
eval_freq = 5000,
n_eval_episodes = 20,
# n_eval_episodes=50,
# For saving a partially trained policy.
eval_target = None,  # Target return value to stop training.
eval_target_n = 2,  # Stop after n consecutive evals above eval_target.
# Agent train args.
initial_explore_steps = 10000,
replay_buffer_size = int(1e6),
model_params = (((200, 200),), 2),
optimizers = (('adam', 0.001),),
batch_size = 256,
weight_decays = (0,),
update_freq = 1,
update_rate = 0.005,
discount = 0.99,

# Construct agent.
agent_flags = utils.Flags(
    action_spec=action_spec,
    observation_spec=observation_spec,
    model_params=model_params,
    optimizers=optimizers,
    batch_size=batch_size,
    weight_decays=weight_decays,
    update_freq=update_freq,
    update_rate=update_rate,
    discount=discount,
    train_data=train_data)
agent_args = agent_module.Config(agent_flags).agent_args
agent = agent_module.Agent(**vars(agent_args))

agent = agents.
agent.restore(agent_ckpt_name)
ckpt_name = './tmp/offlinerl/policies/Instrumental_Variable_Value_Iteration/spede_agent'
# state = utils.load_variable_from_ckpt(ckpt_name, 'state')
print(chkp.print_tensors_in_checkpoint_file(ckpt_name,tensor_name='',all_tensors=True))