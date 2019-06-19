import torch
from torch import nn as nn
from torch.nn import functional as F

from mygan import GANS


class Discriminator(nn.Module):
    def __init__(self, type, hyper, problem):
        super(Discriminator, self).__init__()
        self.ngpu = hyper.ngpu
        self.type = type
        self.main = nn.Sequential(
            nn.Linear(problem.nc, problem.nf),
            nn.LeakyReLU(0.01),
            nn.Linear(problem.nf, problem.nf*2), nn.LeakyReLU(0.01),
            nn.Linear(problem.nf*2, problem.nf*4), nn.LeakyReLU(0.01),
            nn.Linear(problem.nf*4, 1)
        )

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), x.size(1)))
        output = self.main(x)
        if self.type == GANS.GAN :
            output = F.sigmoid(output)
        return output


class Generator(nn.Module):
    def __init__(self, type, hyper, problem):
        super(Generator, self).__init__()
        self.ngpu = hyper.ngpu
        self.type = type
        # self.main = nn.Sequential(nn.Linear(problem.nz, problem.nf), nn.Tanh(), nn.Linear(problem.nf, problem.nc))
        self.main = nn.Sequential(
            nn.Linear(problem.nz, problem.nf), nn.ReLU(),
            nn.Linear(problem.nf, problem.nf * 2), nn.ReLU(),
            nn.Linear(problem.nf * 2, problem.nf * 4), nn.Tanh(), # for CramerGan Tanh shown almost same learnability, fastly detecting more modes, need more comparison to ReLU
            nn.Linear(problem.nf * 4, problem.nc)
        )

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), x.size(1)))
        return self.main(x)