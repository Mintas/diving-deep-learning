from __future__ import print_function
# %matplotlib inline
import random
import torch.nn as nn
import torch.utils.data

import domain.parameters
import multimodalnorm.ganModelMultimodal
import training.losses
import training.optimDecorators
from plots import painters
import mygan
from training import trainer
from multimodalnorm import curvesAndDistributions, splicedNormCurve
import numpy as np

# Set fixed random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 12800  # Batch size during training
nc = 1  # Number of channels in the training images. For color images this is 3| I got 1 channel
imgSize = 1
nz = 1 # Size of z latent vector (i.e. size of generator input) |
ngf = 24  # Size of feature maps in generator | since we generating points it is 2 (x,y)
ndf = 24  # Size of feature maps in discriminator | since we generating points it is 2 (x,y)
num_epochs = 5  # Number of training epochs
lr = 0.0003  # Learning rate for optimizers | 0.04 is good for SGD and 0.0001 for RMSProp
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 0  # Number of GPUs available. Use 0 for CPU mode. | OK I got cpu only
gpWeight = 0.7 # which is good gpWeight? somehow 0.1 is nice, 1 is so so, 10 is bad, 0.01 is vanishing
type = mygan.GANS.CRAMER
initOptimizer = training.optimDecorators.optRMSProp  # works almost as well for SGD and lr = 0.03

m1 = 0
m2 = 7
spread = 4

curve = splicedNormCurve.SplicedNormCurveMany([m1, m2, 14, 20, 30], spread)
curveSample = curve.sampleCurve(batch_size)
curvesAndDistributions.plotPdfAndCdf(curve, batch_size)


# dataSet = curve.sample((128000,))
# np.save('/Users/mintas/PycharmProjects/untitled1/resources/norm07142030', dataSet)

# dataSet = myfuncs.ProbDistrDataset(torch.distributions.normal.Normal(0,1), 128000)
preloaded = np.load('resources/norm07142030.npy')
dataSet = curvesAndDistributions.ProbDistrDataset(curve, 128000, preloaded)

painter = painters.HistorgramPainter(curveSample)
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=1)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
problem = domain.parameters.ProblemSize(nz, ngf, nc, batch_size, imgSize)
hyperParams = domain.parameters.HyperParameters(ngpu, lr, beta1)


def initNet(netClass):
    net = netClass(type, hyperParams, problem).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    net.apply(mygan.weights_init)
    print(net)  # Print the model
    return net

# Initialize BCELoss function; preprocess is my own extension of torch.Dataset
lossCalculator = training.losses.GanLoss(device, problem, nn.BCELoss()) if type == mygan.GANS.GAN \
    else (training.losses.WganLoss if type == mygan.GANS.WGAN else training.losses.CramerGanLoss)(problem, training.losses.GradientPenalizer(gpWeight, True, ngpu > 0))

ganTrainer = trainer.Trainer(device, problem, lossCalculator, initOptimizer, dataSet.preprocess, 'resources/norm07142030.pth')
netG = initNet(multimodalnorm.ganModelMultimodal.Generator)
netD = initNet(multimodalnorm.ganModelMultimodal.Discriminator)

print("Starting Training Loop...")
ganTrainer.train(netD, netG, dataLoader, num_epochs, hyperParams, painter, 12800)
painter.plotFake(netG.forward(torch.randn(128000, nz, 1, 1, device=device)), num_epochs, 0)

painters.plotLosses(ganTrainer.G_losses, ganTrainer.D_losses)
if type != mygan.GANS.GAN :
    painters.plotGradPenalties(ganTrainer.ganLoss.gradientPenalizer.penalties, ganTrainer.ganLoss.gradientPenalizer.norms)
