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
        self.inputSize = problem.nc * (problem.imgSize ** 2)
        nf = problem.nz + problem.nf
        nfx2 = nf * 2
        nfx4 = nf * 4
        self.l1 = nn.Sequential(nn.Linear(self.inputSize + problem.cs, nfx4), nn.LeakyReLU(0.2))
        self.l2 = nn.Sequential(nn.Linear(nfx4 + problem.cs, nfx2), nn.LeakyReLU(0.2))
        self.l3 = nn.Sequential(nn.Linear(nfx2 + problem.cs, nf), nn.LeakyReLU(0.2))
        self.l4 = nn.Sequential(nn.Linear(nf + problem.cs, 1))

    def forward(self, input):
        x = input[0]
        x = x.view(x.shape[0], -1)
         #here, we concat image with condition and pass to FC
        fwd = lambda layer,representation : layer(catGpu(representation, input[1], self.ngpu))
        output = fwd(self.l4, fwd(self.l3, fwd(self.l2, fwd(self.l1, x))))
        if self.type == mygan.GANS.GAN :
            output = F.sigmoid(output)
        return output

class GenEcal(nn.Module):
    def __init__(self, type, hyper, problem):
        super(GenEcal, self).__init__()
        self.ngpu = hyper.ngpu
        self.nc = problem.nc
        self.imgSize = problem.imgSize
        self.inputSize = problem.nz + problem.cs #for generation, concat condition with Z vector
        self.outputSize = problem.nc * (problem.imgSize ** 2)
        nf = problem.nz + problem.nf
        nfx2 = nf * 2
        nfx4 = nf * 4
        self.l1 = nn.Sequential(nn.Linear(problem.nz + problem.cs, nf), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(nf + problem.cs, nfx2), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(nfx2 + problem.cs, nfx4), nn.Tanh())
        self.l4 = nn.Sequential(nn.Linear(nfx4 + problem.cs, self.outputSize), nn.ReLU())

    def forward(self, input):
        z = input[0]
        fwd = lambda layer,representation : layer(catGpu(representation, input[1], self.ngpu))
        imgVector = fwd(self.l4, fwd(self.l3, fwd(self.l2, fwd(self.l1, z))))
        return torch.reshape(imgVector, (z.size(0), self.nc, self.imgSize, self.imgSize))

def catGpu(asArray, condition, ngpu):
    concatenation = torch.cat([asArray, condition], 1)
    if (torch.cuda.is_available() and ngpu > 0) :
        concatenation.cuda()
    return concatenation