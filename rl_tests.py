#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import time
import argparse
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

import hyper_framework

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer

from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.objectives.value import GAE
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="InvertedDoublePendulum-v4")
parser.add_argument("--method", type=str, default="Random") 
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--total_frames", type=int, default=1000000)
parser.add_argument("--t_ready", type=int, default=int(5e4))
parser.add_argument("--folder", type=str, default='temp')

args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cuda:0" if (torch.backends.cuda.is_built()) else "cpu")
num_cells = 256  # number of cells in each layer
lr = 3e-4
max_grad_norm = 1.0

frame_skip = 1
frames_per_batch = 1000 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = args.total_frames // frame_skip

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = GymEnv(args.env, device=device, frame_skip=frame_skip)

if 'v4' in args.env:
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"]),
            StepCounter(),
        ),
    )
else:
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            # DoubleToFloat(in_keys=["observation"]),
            StepCounter(),
        ),
    )

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

check_env_specs(env)

rollout = env.rollout(3)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

policy_module(env.reset())
value_module(env.reset())

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(actor=policy_module,critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)

hyperparameter_bounds = {
    "lambda": [0.9, 1.0],
    "clip_param": [0.1, 0.5],
    "lr": [1e-5, 1e-3],
    "train_batch_size": [int(1e3), int(6e3)],
}

scheduler = hyper_framework.Scheduler(hyperparameter_bounds,args.t_ready,args.method,total_frames)

logs      = defaultdict(list)
logs_eval = defaultdict(list)
# pbar = tqdm(total=total_frames * frame_skip)
eval_str = ""

t0 = time.time()
t1 = time.time()

logdir = "{}_{}_seed{}".format(
        args.env,
        args.method,
        args.seed,
    )

if not os.path.exists(os.path.join(args.folder,logdir)):
    os.makedirs(os.path.join(args.folder,logdir))

writer = SummaryWriter(os.path.join(args.folder,logdir))

savefilepath      = os.path.join(args.folder,logdir,'logs.csv')
eval_savefilepath = os.path.join(args.folder,logdir,'eval_logs.csv')

print(savefilepath)

if os.path.exists(savefilepath) == False:
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    print("Starting Training Loop...")
    for i, tensordict_data in enumerate(tqdm(collector)):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
    
                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()
    
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    
        logs['train_batch_size'].append(sub_batch_size)
        logs['clip_param'].append(loss_module.clip_epsilon.tolist())
        logs['lambda'].append(advantage_module.lmbda.tolist())
        logs['Time'].append(t1-t0)
        logs['Trial'].append('hyperparam_trial')
        logs['Reward'].append(logs['reward'][-1])
        logs['iteration'].append(i)
    
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs_eval["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs_eval["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs_eval["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs_eval['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs_eval['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs_eval['eval step_count'][-1]}"
                )
                del eval_rollout
                
        # Optimizer
        t0 = time.time()
        config_dict = scheduler.step(logs,pd.DataFrame(logs))
        t1 = time.time()
        
        advantage_module.lmbda   = torch.Tensor([config_dict['lambda']])[0].to(device)
        loss_module.clip_epsilon = torch.Tensor([config_dict['clip_param']])[0].to(device)
    
        for g in optim.param_groups:
            g['lr']    = config_dict['lr']
            # g['betas'] = (config_dict['beta1'],config_dict['beta2'])
    
        sub_batch_size = np.ceil(config_dict['train_batch_size']).astype(int)
    
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/train_batch_size'), logs["train_batch_size"][-1])
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/lambda'), logs["lambda"][-1])
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/clip_param'), logs["clip_param"][-1])
    
        writer.add_scalar(os.path.join(args.folder,logdir,'performance/reward'), logs['reward'][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir,'performance/step_count'), logs['step_count'][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir,'hyperparams/learning_rate'), logs["lr"][-1], i)
        
    pd.DataFrame(logs).to_csv(savefilepath)
    pd.DataFrame(logs_eval).to_csv(eval_savefilepath)
    
else:

    print('Log file already exists...passing!')
