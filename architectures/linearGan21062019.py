import torch
import torch.nn as nn
import torch.nn.functional as F
import mygan

# DCGAN, use optAdam or optRMSProp optimizers
#
class DiscEcal(nn.Module):
    def __init__(self, type, hyper, problem):
        super(DiscEcal, self).__init__()
        self.ngpu = hyper.ngpu
        self.type = type
        self.size = problem.nc * (problem.imgSize ** 2)
        nf = problem.nz + problem.nf
        nfx2 = nf * 2
        nfx4 = nf * 4
        self.main = nn.Sequential(
             nn.Linear(self.size, nfx4), nn.LeakyReLU(0.2),
             nn.Linear(nfx4, nfx2), nn.LeakyReLU(0.2),
             nn.Linear(nfx2, nf), nn.LeakyReLU(0.2),
             nn.Linear(nf, 1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self.main(x)
        if self.type == mygan.GANS.GAN :
            output = F.sigmoid(output)
        return output

class GenEcal(nn.Module):
    def __init__(self, type, hyper, problem):
        super(GenEcal, self).__init__()
        self.nc = problem.nc
        self.imgSize = problem.imgSize
        self.size = problem.nc * (problem.imgSize ** 2)
        nf = problem.nz + problem.nf
        nfx2 = nf * 2
        nfx4 = nf * 4
        self.main = nn.Sequential(
            nn.Linear(problem.nz, nf), nn.ReLU(),
            nn.Linear(nf, nfx2), nn.ReLU(),
            nn.Linear(nfx2, nfx4), nn.Tanh(),
            # for CramerGan Tanh shown almost same learnability, fastly detecting more modes, need more comparison to ReLU
            nn.Linear(nfx4, self.size),
            nn.ReLU() # ReLU here, because we need 0+ values
        )


    def forward(self, z):
        z = torch.reshape(z, (z.size(0), z.size(1)))
        imgVector = self.main(z)
        return torch.reshape(imgVector, (z.size(0), self.nc, self.imgSize, self.imgSize))