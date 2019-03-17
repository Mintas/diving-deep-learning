from __future__ import print_function
# %matplotlib inline
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

# Set fixed random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 1280  # Batch size during training
nc = 2  # Number of channels in the training images. For color images this is 3| I got 1 channel
nz = 10  # Size of z latent vector (i.e. size of generator input) |
ngf = 2  # Size of feature maps in generator | since we generating points it is 2 (x,y)
ndf = 2  # Size of feature maps in discriminator | since we generating points it is 2 (x,y)
num_epochs = 222  # Number of training epochs
lr = 0.0005  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 0  # Number of GPUs available. Use 0 for CPU mode. | OK I got cpu only
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

from myfuncs import JoinNormCurve, CurveDataset
import mygan

m1 = 0
m2 = 0
spread = 3
doShuffle = True  # if False will take points without shuffling from 0 to batch_size !
curve = JoinNormCurve(m1, m2, spread)
dataSet = CurveDataset(128000, curve.curve, curve.interval)
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=doShuffle, num_workers=1)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
problem = mygan.ProblemSize(nz, ngf, nc)
hyperParams = mygan.HyperParameters(ngpu, lr, beta1)


def initNet(netClass):
    net = netClass(hyperParams, problem).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    net.apply(mygan.weights_init)
    print(net)  # Print the model
    return net

netG = initNet(mygan.Generator)
netD = initNet(mygan.Discriminator)

criterion = nn.BCELoss()  # Initialize BCELoss function

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(256, nz, 1, 1, device=device)

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
rlabel = torch.full(((batch_size),), real_label, device=device)
flabel = torch.full(((batch_size),), fake_label, device=device)

iters = 0

def fwdBwdError(net, input, labels):
    # Forward pass real batch through D
    output = net(input).view(-1)
    # Calculate loss on all-real batch
    err = criterion(output, labels)
    # Calculate gradients for D in backward pass
    err.backward()
    return err, output.mean().item()

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataLoader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        bbbb = torch.from_numpy(np.column_stack((data[0], data[1]))).view(batch_size, nc, 1, 1).to(device)
        errD_real, D_x = fwdBwdError(netD, bbbb, rlabel)

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D         # Calculate D's loss on the all-fake batch         # Calculate the gradients for this batch
        errD_fake, D_G_z1 = fwdBwdError(netD, fake.detach(), flabel)

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        netD.optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        errG, D_G_z2 = fwdBwdError(netD, fake, rlabel)         # fake labels are real for generator cost
        # Update G
        netG.optimizer.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataLoader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataLoader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(plt.figure())
            reshape = fake.reshape((fake.size(0), fake.size(1)))
            sorted = reshape[reshape[:,0].argsort()]
            reshaped = torch.t(sorted)
            asList = reshaped.tolist()
            plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
            plt.plot(asList[0], asList[1])

        iters += 1


def plotLosses():
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()