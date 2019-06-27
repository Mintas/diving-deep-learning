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
        nfx2 = problem.nf * 2
        nfx4 = problem.nf * 4
        kernel_size = 4
        stride = 2
        self.main = nn.Sequential(
             nn.Conv2d(problem.nc, problem.nf, 2, stride, 1, bias=False), nn.LeakyReLU(0.2),
             nn.Conv2d(problem.nf, nfx2, kernel_size, stride, 1, bias=False),
             nn.LeakyReLU(0.2),
             nn.Conv2d(nfx2, 1, kernel_size, stride, 1, bias=False)
        )

    def forward(self, x):
        output = self.main(x)
        if self.type == mygan.GANS.GAN :
            output = F.sigmoid(output)
        return output

class GenEcal(nn.Module):
    def __init__(self, type, hyper, problem):
        super(GenEcal, self).__init__()
        self.ngpu = hyper.ngpu
        kernel_size = 4
        stride = 2
        self.nfx4 = problem.nf * 4
        self.nfx2 = problem.nf * 2
        self.main = nn.Sequential(
            #output 120x4x4
            nn.ConvTranspose2d(problem.nz, self.nfx2, kernel_size, stride, 1, bias=False), #nn.BatchNorm2d(self.nfx2),
            nn.ReLU(),
            #output 30x16x16
            nn.ConvTranspose2d(self.nfx2, problem.nf, kernel_size, stride, 1, bias=False), #nn.BatchNorm2d(problem.nf),
            nn.ReLU(),
            nn.ConvTranspose2d(problem.nf, problem.nc, 2, stride, 0, bias=False),
            nn.ReLU() # ReLU here, because we need 0+ values
        )


    def forward(self, x):
        return self.main(x)

#import torch
#z = torch.rand((100)).view(1, 100, 1, 1)
#netG = GenEcal(mygan.GANS.WGAN, p.HyperParameters(0, 1, 1), p.ProblemSize(100, 30, 1, 1, 8))
#outG = netG(z)