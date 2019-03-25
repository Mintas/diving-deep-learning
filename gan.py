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
nc = 1  # Number of channels in the training images. For color images this is 3| I got 1 channel
nz = 1 # Size of z latent vector (i.e. size of generator input) |
ngf = 10  # Size of feature maps in generator | since we generating points it is 2 (x,y)
ndf = 10  # Size of feature maps in discriminator | since we generating points it is 2 (x,y)
num_epochs = 400  # Number of training epochs
lr = 0.04  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 0  # Number of GPUs available. Use 0 for CPU mode. | OK I got cpu only
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

from myfuncs import SplicedNormCurve, CurveDataset, ProbDistrDataset, FromPdfDistribution
import mygan

m1 = 0
m2 = 0
spread = 3
doShuffle = True  # if False will take points without shuffling from 0 to batch_size !
curve = SplicedNormCurve(m1, m2, spread)
curveSample = curve.sampleCurve(batch_size)

splicedDistr = FromPdfDistribution(curve.curve)
# points = splicedDistr.rvs(size=128000)
# dataSet = CurveDataset(128000, curve.curve, curve.interval)
dataSet = ProbDistrDataset(torch.distributions.normal.Normal(0,1), 128000)
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=doShuffle, num_workers=1)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
problem = mygan.ProblemSize(nz, ngf, nc)
hyperParams = mygan.HyperParameters(ngpu, batch_size, lr, beta1)


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
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
rlabel = torch.full(((batch_size),), real_label, device=device)
flabel = torch.full(((batch_size),), fake_label, device=device)


def fwdBwdError(net, input, labels):
    # Forward pass real batch through D
    output = net(input).view(-1)
    # Calculate loss on all-real batch
    err = criterion(output, labels)
    # Calculate gradients for D in backward pass
    err.backward()
    return err, output.mean().item()

def plotLosses():
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

print("Starting Training Loop...")
iters = 0


def plotCurve(fake, data):
    reshape = fake.reshape((fake.size(0), fake.size(1)))
    sorted = reshape[reshape[:, 0].argsort()]
    reshaped = torch.t(sorted)
    asList = reshaped.tolist()
    plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
    plt.plot(asList[0], asList[1], label="FromNoise")
    plt.plot(np.sort(data[0]), data[1][np.argsort(data[0])].numpy(), label="sampledBatch")
    plt.show()


def prepareCurveData(data):
    return np.column_stack((data[0], data[1]))


# For each epoch
def plotHist(fake, data):
    fake = fake.reshape(fake.size(0))
    maxBins = 50
    ranges = [min(min(fake), min(data)), max(max(fake), max(data))]
    binWidth = (ranges[1] - ranges[0]) / maxBins
    bins = np.arange(ranges[0], ranges[1] + binWidth, binWidth)

    _, ax = plt.subplots()
    ax.hist(fake, bins=bins, alpha=1, density=True, label='fake generated')
    ax.hist(data, bins=bins, alpha=0.75, density=True, label='data sampled')
    plt.plot(curveSample[0], curveSample[1])
    plt.show()


for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataLoader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        # plt.plot(np.ndarray(datacopy[0]), np.ndarray(datacopy[1]))

        # prepared = torch.from_numpy(prepareCurveData(data))
        prepared = data
        bbbb = prepared.view(batch_size, nc, 1, 1).to(device)
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
        D_losses.append(errD.item())

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % 2 == 0 :
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            errG, D_G_z2 = fwdBwdError(netD, fake, rlabel)         # fake labels are real for generator cost
            # Update G
            netG.optimizer.step()
        #hint ?
        G_losses.append(errG.item())

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataLoader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later


        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataLoader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(plt.figure())
            # plotCurve(fake, data)
            plotHist(fake, data)
        iters += 1
