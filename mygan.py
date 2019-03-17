import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


class ProblemSize:
    def __init__(self, latentInputLength, featureSize, channelsDepth):
        self.nz = latentInputLength
        self.nf = featureSize
        self.nc = channelsDepth


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, ngpu, problem):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nfx2 = problem.nf * 2
        kernel_size = 3
        stride = 1
        self.main = nn.Sequential(
            nn.Conv2d(problem.nc, nfx2, kernel_size, stride, 1, bias=False),  # input is 2 x 1 x 1, output of 4x1x1
            nn.LeakyReLU(0.2),
            nn.Conv2d(nfx2, 1, kernel_size, stride, 1, bias=False),  # output an answers real/fake
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, ngpu, problem):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        kernel_size = 3
        stride = 1
        self.nfx4 = problem.nf * 4
        self.nfx2 = problem.nf * 2
        self.main = nn.Sequential(
            nn.ConvTranspose2d(problem.nz, self.nfx4, 1, stride, 0, bias=False), nn.BatchNorm2d(self.nfx4), nn.ReLU(),
            nn.ConvTranspose2d(self.nfx4, self.nfx2, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(self.nfx2), nn.ReLU(),
            nn.ConvTranspose2d(self.nfx2, problem.nf, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(problem.nf), nn.ReLU(),
            nn.ConvTranspose2d(problem.nf, problem.nc, 1, stride, 0, bias=False), nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
