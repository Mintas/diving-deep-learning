from __future__ import print_function
# %matplotlib inline
import random
import torch
import torch.nn as nn
import torch.utils.data

import domain.ecaldata
import domain.parameters

import training.losses
import training.optimDecorators
from plots import painters, plotUi
import mygan
import architectures.dcganBatchNorm as myzoo
from training import trainer
import numpy as np
from serialization import iogan

from analytics import analytic_funcs as AF
from analytics import optimized_analytic_funcs as OAF
from os.path import dirname



# Set fixed random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

LossDict = {mygan.GANS.GAN : training.losses.GanLoss,
            mygan.GANS.WGAN : training.losses.WganLoss,
            mygan.GANS.CRAMER : training.losses.CramerGanLoss,
            mygan.GANS.ECRAMER : training.losses.CramerEneryGanLoss}


batch_size = 500  # Batch size during training
nc = 1  # we got 1channel response
nz = 100 # latent space size | 42 is close to hypotenuse of response 30x30
imgSize = 30  # our respons is 30x30
ngf = 100  # todo : decide Generator feature-space characteristic size
ndf = 30  # decide Critic feature-space characteristic size
num_epochs = 5  # 5 for example, need much more for learning
lr = 0.0003  # Learning rate for optimizers | 0.04 is good for SGD and 0.0001 for RMSProp
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 0  # increase for GPU hosted calculations
gpWeight = 0.5 # gradient penalty weight; somehow 0.1 is nice, 1 is so so, 10 is bad, 0.01 is vanishing
type = mygan.GANS.CRAMER # we are going to try gan, wgan-gp and cramerGan
initOptimizer = training.optimDecorators.optRMSProp  # works almost as well for SGD and lr = 0.03

# dataSet = myfuncs.ProbDistrDataset(torch.distributions.normal.Normal(0,1), 128000)
datasetName = 'caloGAN_v4_case2_50K'
archVersion = 'dcganZMatch' #arch version

resultingName = 'resources/computed/%s_%s' % (datasetName, archVersion)
ganFile = resultingName
pdfFile = resultingName #pdf is added in PDFPlotUi
statFile = resultingName
#ecalData = np.load('ecaldata/caloGAN_v3_case4_2K.npz')

# EnergyDeposit = ecal['EnergyDeposit']
# ParticlePoint = ecal['ParticlePoint']
# ParticleMomentum = ecal['ParticleMomentum']
# ParticlePDG=ecal['ParticlePDG']  #here we got only vector with constant 22




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


netG = initNet(myzoo.GenEcal)
netD = initNet(myzoo.DiscEcal)

print(netG)
print(netD)


def trainGan():
    ecalData = np.load('resources/ecaldata/%s.npz' % datasetName)

    dataSet = torch.utils.data.TensorDataset(torch.from_numpy(ecalData['EnergyDeposit']).float())
    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=True, num_workers=1)

    lossCalculator = training.losses.GanLoss(device, problem, nn.BCELoss()) if type == mygan.GANS.GAN \
        else (training.losses.WganLoss if type == mygan.GANS.WGAN else training.losses.CramerEneryGanLoss)(problem,
                                                                                                           training.losses.GradientPenalizer(gpWeight,True,ngpu > 0))

    ganTrainer = trainer.Trainer(device, problem, lossCalculator, initOptimizer, lambda d: d[0], ganFile)

    ui = plotUi.ShowPlotUi()
    painter = painters.ECalPainter(ui)
    print("Starting Training Loop...")
    dopt, gopt = ganTrainer.train(netD, netG, dataLoader, num_epochs, hyperParams, painter, 9)
    # painter.plotFake(netG.forward(torch.randn(128000, nz, 1, 1, device=device)), num_epochs, 0)

    ui.toView(lambda: painters.plotLosses(ganTrainer.G_losses, ganTrainer.D_losses))

    if type != mygan.GANS.GAN:
        ui.toView(lambda: painters.plotGradPenalties(ganTrainer.ganLoss.gradientPenalizer.penalties,
                                                     ganTrainer.ganLoss.gradientPenalizer.norms))

    ui.close()

def evalGan():
    with torch.no_grad():
        #iogan.loadGANs(netD, netG, ganFile)
        checkpoint = torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_dcganBatchNorm_nf100 (2).pth', map_location='cpu')

        netD.load_state_dict(checkpoint[iogan.IOGANConst.D])
        netG.load_state_dict(checkpoint[iogan.IOGANConst.G])
        netG.eval(), netD.eval()

        ecalData = domain.ecaldata.parseEcalData(datasetName)

        shape = ecalData.response.shape
        print(shape)

        tonnsOfNoise = torch.randn(shape[0], nz, 1, 1, device=device)
        generated = netG(tonnsOfNoise)
        responses = generated.reshape(shape).cpu().detach().numpy()
        fakeData = domain.ecaldata.EcalData(responses, ecalData.momentum, ecalData.point)

        #OAF.runAnalytics(statFile, ecalData, fakeData)
        es, fs = OAF.runAnalytics('/' + 'caloGAN_v4_case2_50K_dcganBatchNprm', ecalData, fakeData,
                         ecalStats=torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_stats.pth'))


#trainGan()
evalGan()
