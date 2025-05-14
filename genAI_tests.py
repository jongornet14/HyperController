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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


import hyper_framework

from collections import defaultdict

import matplotlib.pyplot as plt
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tqdm import tqdm

import torch.multiprocessing as mp

# python3 genAI_tests.py --seed 0 --method 'Random' --folder 'results' --home_dir='/home/jonathangornet/Documents/' --data_dir='ML_datasets/celebA/' --dataset='celebA'

parser = argparse.ArgumentParser()
# parser.add_argument("--method", type=str, default="Random") 
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t_ready", type=int, default=int(400))
parser.add_argument("--folder", type=str, default='temp')
parser.add_argument("--method", type=str, default="Random") 
parser.add_argument("--dataset", type=str, default='cifar10')
parser.add_argument("--data_dir", type=str, default='datasets/cifar10/images')
parser.add_argument("--home_dir", type=str, default='/home/jonathangornet/Documents/')

args = parser.parse_args()

torch.manual_seed(args.seed)
# torch.use_deterministic_algorithms(True) # Needed for reproducible results



method       = args.method
dataset_name = args.dataset
# Root directory for dataset
dataroot     = '/home/jonathangornet/Documents/ML_datasets/' + str(dataset_name) + '/'#args.home_dir + args.data_dir

# Dataset and Spatial size of training images. All images will be resized to this
#   size using a transformer.
if dataset_name=='cifar10':#args.dataset == 'cifar10':
    image_size = 32
    # Size of feature maps in generator
    ngf = 32
    
    # Size of feature maps in discriminator
    ndf = 32
else:
    image_size = 64
    # Size of feature maps in generator
    ngf = 64
    
    # Size of feature maps in discriminator
    ndf = 64
    
print(method)
print(dataset_name)

# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.backends.cuda.is_built() and ngpu > 0) else "cpu")


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=workers)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

logs = defaultdict(list)

t0 = time.time()
t1 = time.time()

logdir = "{}_{}_seed{}".format(
        dataset_name,
        method,
        args.seed,
    )

if not os.path.exists(os.path.join(args.folder,logdir)):
    os.makedirs(os.path.join(args.folder,logdir))

writer = SummaryWriter(os.path.join(args.folder,logdir))

#------------------------------------ Scheduler ------------------------------------

hyperparameter_bounds = {
    "g_lr": [1e-5, 1e-3],
    "d_lr": [1e-5, 1e-3],
    "g_beta1": [0.5, 0.9],
    "g_beta2": [0.5, 0.999],
    # "g_weight_decay": [1e-5,1e-3],
    "d_beta1": [0.5, 0.9],
    "d_beta2": [0.5, 0.999],
    # "d_weight_decay": [1e-5,1e-3],
}

scheduler = hyper_framework.Scheduler(hyperparameter_bounds,args.t_ready,method)

print("Starting Training Loop...")

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        #------------------------------------ Hyperparams ------------------------------------

        logs["g_loss"].append(errG.item())
        logs["d_loss"].append(errD.item())

        logs["Reward"].append( - errG.item() - errD.item())
        logs["reward"].append( - errG.item() - errD.item())
        
        logs["g_lr"].append(optimizerG.param_groups[0]["lr"])
        logs["d_lr"].append(optimizerD.param_groups[0]["lr"])

        logs["g_weight_decay"].append(optimizerG.param_groups[0]["weight_decay"])
        logs["d_weight_decay"].append(optimizerD.param_groups[0]["weight_decay"])

        logs["g_beta1"].append(optimizerG.param_groups[0]["betas"][0])
        logs["d_beta1"].append(optimizerD.param_groups[0]["betas"][0])

        logs["g_beta2"].append(optimizerG.param_groups[0]["betas"][1])
        logs["d_beta2"].append(optimizerD.param_groups[0]["betas"][1])
    
        logs['Time'].append(t1-t0)
        logs['Trial'].append('hyperparam_trial')
        logs['iteration'].append(iters)

        # Optimizer
        t0 = time.time()
        config_dict = scheduler.step(logs,pd.DataFrame(logs))
        t1 = time.time()
        
        for g in optimizerG.param_groups:
            g['lr']           = config_dict['g_lr']
            g['betas']        = (config_dict['g_beta1'],config_dict['g_beta2'])
            # g['weight_decay'] = config_dict['g_weight_decay']
        for g in optimizerD.param_groups:
            g['lr']           = config_dict['d_lr']
            g['betas']        = (config_dict['d_beta1'],config_dict['d_beta2'])
            # g['weight_decay'] = config_dict['d_weight_decay']

        writer.add_scalar(os.path.join(args.folder,logdir,'performance/reward'), logs['reward'][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/g_learning_rate'), logs["g_lr"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/d_learning_rate'), logs["d_lr"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/g_beta1'), logs["g_beta1"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/d_beta1'), logs["d_beta1"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/g_beta2'), logs["g_beta2"][-1], i)
        writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/d_beta2'), logs["d_beta2"][-1], i)
        # writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/g_weight_decay'), logs["g_weight_decay"][-1], i)
        # writer.add_scalar(os.path.join(args.folder,logdir, 'hyperparams/d_weight_decay'), logs["d_weight_decay"][-1], i)

        iters += 1

savefilepath = os.path.join(args.folder,logdir,'logs.csv')

pd.DataFrame(logs).to_csv(savefilepath)