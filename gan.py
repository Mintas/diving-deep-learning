from __future__ import print_function
# %matplotlib inline
import random
import torch
import torch.nn as nn
import torch.utils.data
import painters
import mygan
import trainer
import myfuncs
import myfuncsnew
import numpy as np

# Set fixed random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 1280  # Batch size during training
nc = 1  # Number of channels in the training images. For color images this is 3| I got 1 channel
nz = 1 # Size of z latent vector (i.e. size of generator input) |
ngf = 16  # Size of feature maps in generator | since we generating points it is 2 (x,y)
ndf = 16  # Size of feature maps in discriminator | since we generating points it is 2 (x,y)
num_epochs = 500  # Number of training epochs
lr = 0.0003  # Learning rate for optimizers | 0.04 is good for SGD and 0.0001 for RMSProp
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 0  # Number of GPUs available. Use 0 for CPU mode. | OK I got cpu only
gpWeight = 0.7 # which is good gpWeight? somehow 0.1 is nice, 1 is so so, 10 is bad, 0.01 is vanishing
type = mygan.GANS.CRAMER
initOptimizer = mygan.optRMSProp # works almost as well for SGD and lr = 0.03

m1 = 0
m2 = 7
spread = 4

curve = myfuncsnew.SplicedNormCurveMany([m1,m2, 14], spread)
curveSample = curve.sampleCurve(batch_size)
myfuncs.plotPdfAndCdf(curve, batch_size)


# dataSet = curve.sample((128000,))
# np.save('/Users/mintas/PycharmProjects/untitled1/resources/norm0714', dataSet)

# dataSet = myfuncs.ProbDistrDataset(torch.distributions.normal.Normal(0,1), 128000)
preloaded = np.load('resources/norm0714.npy')
dataSet = myfuncs.ProbDistrDataset(curve, 128000, preloaded)

painter = painters.HistorgramPainter(curveSample)
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=1)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
problem = mygan.ProblemSize(nz, ngf, nc, batch_size)
hyperParams = mygan.HyperParameters(ngpu, lr, beta1)


def initNet(netClass):
    net = netClass(type, hyperParams, problem).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    net.apply(mygan.weights_init)
    print(net)  # Print the model
    return net

# Initialize BCELoss function; preprocess is my own extension of torch.Dataset
lossCalculator = trainer.GanLoss(device, problem, nn.BCELoss()) if type == mygan.GANS.GAN \
    else (trainer.WganLoss if type == mygan.GANS.WGAN else trainer.CramerGanLoss)(problem, mygan.GradientPenalizer(gpWeight, True, ngpu > 0))

ganTrainer = trainer.Trainer(device, problem, lossCalculator, initOptimizer, dataSet.preprocess)
netG = initNet(mygan.Generator)
netD = initNet(mygan.Discriminator)

print("Starting Training Loop...")
ganTrainer.train(netD, netG, dataLoader, num_epochs, hyperParams, painter, 12800)
painter.plotFake(netG.forward(torch.randn(128000, nz, 1, 1, device=device)), num_epochs, 0)

painters.plotLosses(ganTrainer.G_losses, ganTrainer.D_losses)
if type != mygan.GANS.GAN :
    painters.plotGradPenalties(ganTrainer.ganLoss.gradientPenalizer.penalties, ganTrainer.ganLoss.gradientPenalizer.norms)
