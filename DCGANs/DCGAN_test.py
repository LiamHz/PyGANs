from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import time
import pickle
import math
from DCGAN_models import Generator

# Set random seed for reproducibility
# manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

animate_result = True

# Detemines what type of GAN model will be trained
gan_type = 'cat'

# Location of data on disk, disk location of models
# and root directory for dataset
if gan_type == 'celeb':
    PATH_TO_DATA = './models/celeb_dcgan/celeb_data.pkl'
    PATH_TO_LOAD_MODEL_G = './models/celeb_dcgan/celeb_G_Model'
    PATH_TO_IMAGE_LOGS = './celeb_gans_images'
elif gan_type == 'dog':
    PATH_TO_DATA = './models/dog_dcgan/dog_data.pkl'
    PATH_TO_LOAD_MODEL_G = './models/dog_dcgan/dog_G_Model'
    PATH_TO_IMAGE_LOGS = './dog_gans_images'
elif gan_type == 'cat':
    PATH_TO_DATA = './models/cat_dcgan/cat_data.pkl'
    PATH_TO_LOAD_MODEL_G = './models/cat_dcgan/cat_G_Model'
    PATH_TO_IMAGE_LOGS = './cat_gans_images'
elif gan_type == 'pokemon':
    PATH_TO_DATA = './models/pokemon_dcgan/pokemon_data.pkl'
    PATH_TO_LOAD_MODEL_G = './models/pokemon_dcgan/pokemon_G_Model'
    PATH_TO_IMAGE_LOGS = './pokemon_gans_images'

# Parameters for G
# Number of channels in the training images. For color images this is 3 (RGB)
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Number of GPUs available. Use 0 for CPU mode
ngpu = 1

# Number of fake images to display at once
num_images_display = 4

print("\nGAN for", gan_type)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.load_state_dict(torch.load(PATH_TO_LOAD_MODEL_G + '.pt'))
print("\nModel G loaded from disk")

print("Creating animation")

fake_images = []

# Generate fake images
with torch.no_grad():
    noise = torch.randn(num_images_display, nz, 1, 1, device=device)

    for i in range(1000):
        # Change fake image slightly

        # List of indicies to edit in noise
        rand_index = random.randint(0, (nz-1))
        rand_index2 = (rand_index + 25) % nz
        if rand_index2 < rand_index:
            rand_index, rand_index2 = rand_index2, rand_index

        # Edit value on each image
        for j in range(num_images_display):
            noise[j][rand_index : rand_index2] += (random.randint(-25, 25) / 10)

        fake = netG(noise).detach().cpu()
        fake_images.append(vutils.make_grid(fake, padding=2, normalize=True))

sample_frequency = 10
display_images = []
display_images = fake_images[::sample_frequency]

print("Displaying generated images")

# Animate fake images
fig = plt.figure(figsize=(2,2))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in display_images]
ani = animation.ArtistAnimation(fig, ims, interval=2000, repeat_delay=2000, blit=False)

HTML(ani.to_jshtml())
plt.show()
