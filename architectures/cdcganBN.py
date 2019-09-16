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
        self.imgSize = problem.imgSize
        nfx2 = problem.nf * 2
        nfx4 = problem.nf * 4
        kernel_size = 4
        stride = 2
        self.main = nn.Sequential(
             nn.Conv2d(problem.nc + problem.cs, problem.nf, 2, stride, 1, bias=False), nn.BatchNorm2d(problem.nf),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(problem.nf, nfx2, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(nfx2),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(nfx2, nfx4, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(nfx4),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(nfx4, 1, kernel_size, 1, 0, bias=False)
        )

    def forward(self, input):
        condition = input[1]
        #bs = condition.size(0)
        #cs = condition.size(1)
        #conditionAsChannels = condition.unsqueeze(-1).expand(bs, cs, self.imgSize).unsqueeze(-1).expand(bs, cs, self.imgSize, self.imgSize)
        x = torch.cat([input[0], condition], 1)
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
            nn.ConvTranspose2d(problem.nz + problem.cs, self.nfx4, kernel_size, 1, 0, bias=False), nn.BatchNorm2d(self.nfx4),
            nn.ReLU(True),
            #output 60x8x8
            nn.ConvTranspose2d(self.nfx4, self.nfx2, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(self.nfx2),
            nn.ReLU(True),
            #output 30x16x16
            nn.ConvTranspose2d(self.nfx2, problem.nf, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(problem.nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(problem.nf, problem.nc, 2, stride, 1, bias=False),
            nn.ReLU(True) # ReLU here, because we need 0+ values
        )

    def forward(self, input):
        x = torch.cat([input[0], input[1]], 1)
        x = x.reshape((x.size(0), x.size(1), 1, 1))
        return self.main(x)