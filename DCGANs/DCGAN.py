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
from DCGAN_models import Discriminator

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Determines if model will animate progress
# If false, progress pics will each be put on a seperate plot
animate_progress = False

# Determines if models are loaded from disk
load_models_from_disk = True

# Determines if data (img_list, G_losses, D_losses, and iters) is loaded from disk
load_data_from_disk = True

# Determines if model will train, if false, model will be set for evaluation
train_model = True

# Detemines what type of GAN model will be trained
gan_type = 'dog'

# Number of training epochs
# Specifies how many additional epochs to run
num_epochs = 70

# Location of data on disk, disk location of models
# and root directory for dataset
if gan_type == 'celeb':
    PATH_TO_DATA = './models/celeb_dcgan/celeb_data.pkl'
    PATH_TO_DATA_BACKUP = './models/dog_dcgan/backups/dog_data'
    PATH_TO_LOAD_MODEL_D = './models/celeb_dcgan/celeb_D_Model'
    PATH_TO_LOAD_MODEL_G = './models/celeb_dcgan/celeb_G_Model'
    PATH_TO_SAVE_BACKUP_MODEL_D = './models/celeb_dcgan/backups/celeb_D_Model'
    PATH_TO_SAVE_BACKUP_MODEL_G = './models/celeb_dcgan/backups/celeb_G_Model'
    PATH_TO_SAVE_MODEL_D = './models/celeb_dcgan/celeb_D_Model'
    PATH_TO_SAVE_MODEL_G = './models/celeb_dcgan/celeb_G_Model'
    PATH_TO_IMAGE_LOGS = './celeb_gans_images'
    dataroot = r"C:\Users\liamh\Documents\datasets\celeba"
elif gan_type == 'dog':
    PATH_TO_DATA = './models/dog_dcgan/dog_data.pkl'
    PATH_TO_DATA_BACKUP = './models/dog_dcgan/backups/dog_data'
    PATH_TO_LOAD_MODEL_D = './models/dog_dcgan/dog_D_Model'
    PATH_TO_LOAD_MODEL_G = './models/dog_dcgan/dog_G_Model'
    PATH_TO_SAVE_BACKUP_MODEL_D = './models/dog_dcgan/backups/dog_D_Model'
    PATH_TO_SAVE_BACKUP_MODEL_G = './models/dog_dcgan/backups/dog_G_Model'
    PATH_TO_SAVE_MODEL_D = './models/dog_dcgan/dog_D_Model'
    PATH_TO_SAVE_MODEL_G = './models/dog_dcgan/dog_G_Model'
    PATH_TO_IMAGE_LOGS = './dog_gans_images'
    dataroot = r"C:\Users\liamh\Documents\datasets\cats-and-dogs\dogs"
elif gan_type == 'cat':
    PATH_TO_DATA = './models/cat_dcgan/cat_data.pkl'
    PATH_TO_DATA_BACKUP = './models/cat_dcgan/backups/cat_data'
    PATH_TO_LOAD_MODEL_D = './models/cat_dcgan/cat_D_Model'
    PATH_TO_LOAD_MODEL_G = './models/cat_dcgan/cat_G_Model'
    PATH_TO_SAVE_BACKUP_MODEL_D = './models/cat_dcgan/backups/cat_D_Model'
    PATH_TO_SAVE_BACKUP_MODEL_G = './models/cat_dcgan/backups/cat_G_Model'
    PATH_TO_SAVE_MODEL_D = './models/cat_dcgan/cat_D_Model'
    PATH_TO_SAVE_MODEL_G = './models/cat_dcgan/cat_G_Model'
    PATH_TO_IMAGE_LOGS = './cat_gans_images'
    dataroot = r"C:\Users\liamh\Documents\datasets\cats_annotated"
elif gan_type == 'pokemon':
    PATH_TO_DATA = './models/pokemon_dcgan/pokemon_data.pkl'
    PATH_TO_DATA_BACKUP = './models/pokemon_dcgan/backups/pokemon_data'
    PATH_TO_LOAD_MODEL_D = './models/pokemon_dcgan/pokemon_D_Model'
    PATH_TO_LOAD_MODEL_G = './models/pokemon_dcgan/pokemon_G_Model'
    PATH_TO_SAVE_BACKUP_MODEL_D = './models/pokemon_dcgan/backups/pokemon_D_Model'
    PATH_TO_SAVE_BACKUP_MODEL_G = './models/pokemon_dcgan/backups/pokemon_G_Model'
    PATH_TO_SAVE_MODEL_D = './models/pokemon_dcgan/pokemon_D_Model'
    PATH_TO_SAVE_MODEL_G = './models/pokemon_dcgan/pokemon_G_Model'
    PATH_TO_IMAGE_LOGS = './pokemon_gans_images'
    dataroot = r"C:\Users\liamh\Documents\datasets\pokemon"

# Number of workers for dataloader (Image loading)
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images.
# All images will be resized to this size using a transformer
image_size = 64

# Number of channels in the training images. For color images this is 3 (RGB)
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in generator
ndf = 64

# Learning rate for optimizers
lr_g = 0.0002
if gan_type == 'cat' or gan_type == 'dog':
    # 1/4 of DCGAN paper original value, recommended by AlexiaJM
    lr_d = 0.00005
else:
    lr_d = 0.0002

# Beta1 heperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode
ngpu = 1

# We can use an image folder dataset the way we have it setup
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

# Create the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


if __name__ == '__main__':      # Windows PyTorch multiprocessing support
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    # Custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    print("\nGAN for", gan_type)

    # Create the generator
    netG = Generator(ngpu, nz, ngf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2
    if load_models_from_disk:
        netG.load_state_dict(torch.load(PATH_TO_LOAD_MODEL_G + '.pt'))
        print("\nModel G loaded from disk")
    else:
        netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu, nz, ndf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))


    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    if load_models_from_disk:
        netD.load_state_dict(torch.load(PATH_TO_LOAD_MODEL_D + '.pt'))
        print("\nModel D loaded from disk")
    else:
        netD.apply(weights_init)

    # Print the model
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels deuring training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))


    # Training Loop

    # Lists to keep track of progress
    if load_data_from_disk:
        pickle_jar = open(PATH_TO_DATA, 'rb')
        data = pickle.load(pickle_jar)
        img_list = data['img_list']
        G_losses = data['G_losses']
        D_losses = data['D_losses']
        iters = data['iters']
        epoch = data['epoch']
        if load_models_from_disk:
            # num_epochs specifies how many additional epochs to run
            num_epochs += epoch
        print('\nData loaded from disk')
        print('epoch:', data['epoch'])
        print('num_epochs:', num_epochs)
    else:
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        epoch = 1

    if train_model:
        print("Starting Training Loop...")
        training_start_time = time.time()
        # For each epoch
        while epoch <= num_epochs:
            epoch_start_time = time.time()
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
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate D's loss on all-real batch
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
                # Classify all fake batches with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)     # fake labels are real for generator cost
                # Since D was just updated, perform another forward pass of all-fake batch through D
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

                # Check how the generator is doing by saving G's output
                # on fixed_noise once per epoch
                if (i == len(dataloader)-1):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            # Benchmarking stats
            epoch_end_time = time.time()
            print("Epoch", epoch, "Time Elapsed:", str(math.floor(epoch_end_time - epoch_start_time)) + "s\n")
            print("Estimated time remaining:", str(math.floor((num_epochs - epoch) * (epoch_end_time - epoch_start_time) / 60)) + "m\n" )

            # Save backups once every 10 epochs
            if epoch % 10 == 0 or gan_type == 'celeb':
                # Backup model to disk
                torch.save(netD.state_dict(), PATH_TO_SAVE_BACKUP_MODEL_D + '_epoch_' + str(epoch) + '.pt')
                torch.save(netG.state_dict(), PATH_TO_SAVE_BACKUP_MODEL_G + '_epoch_' + str(epoch) + '.pt')
                print("Backup: Models for D and G to disk")

                # Backup data to disk
                data = {}
                data['img_list'] = img_list
                data['G_losses'] = G_losses
                data['D_losses'] = D_losses
                data['iters'] = iters
                data['epoch'] = epoch + 1
                pickle_jar = open(PATH_TO_DATA_BACKUP + '_epoch_' + str(epoch) + '.pkl', 'wb')
                pickle.dump(data, pickle_jar)
                pickle_jar.close()
                print('Backup: Data saved to disk\n')

            epoch += 1

        # Save model at the end of a training session to disk
        torch.save(netD.state_dict(), PATH_TO_SAVE_MODEL_D + '.pt')
        torch.save(netG.state_dict(), PATH_TO_SAVE_MODEL_G + '.pt')
        print("Models for D and G saved to disk")

        # Save data at the end of a training session to disk
        data = {}
        data['img_list'] = img_list
        data['G_losses'] = G_losses
        data['D_losses'] = D_losses
        data['iters'] = iters
        data['epoch'] = epoch
        pickle_jar = open(PATH_TO_DATA, 'wb')
        pickle.dump(data, pickle_jar)
        pickle_jar.close()
        print('Data saved to disk')

        training_end_time = time.time()
        print("Training Session Time Elapsed:", str(math.floor((training_end_time - training_start_time) / 60)) + "m\n")
        approx_total_training_time = epoch_end_time - epoch_start_time
        print("Total Model Training Time:", str(math.floor((epoch_end_time - epoch_start_time) * epoch / 60)) + "m\n")
    else:
        # Prep G and D for evalutation
        netD.eval()
        netG.eval()

    # Plot loss versus training iteration
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(PATH_TO_IMAGE_LOGS + '/G_and_D_loss_during_training.png', bbox_inches="tight")

    # Visualize training progress of G
    if animate_progress:
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())
    else:
        # Save progress pics to disk
        for index, img in enumerate(img_list):
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.imshow(np.transpose(img,(1,2,0)))
            plt.savefig(PATH_TO_IMAGE_LOGS + '/progress_pics/progress_'+str(index)+'.png', bbox_inches="tight")
            plt.close()     # Prevents flood of progress pics filling display


    # Display real and fake images side by side
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1,2,0)))
    plt.savefig(PATH_TO_IMAGE_LOGS + '/real_vs_fake.png', bbox_inches="tight")
    plt.show()
