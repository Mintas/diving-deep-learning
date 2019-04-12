import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim


class GANS :
    GAN = 'GAN'
    WGAN = 'WGAN'
    CRAMER = 'CRAMER'

class ProblemSize:
    def __init__(self, latentInputLength, featureSize, channelsDepth,  batchSize):
        self.nz = latentInputLength
        self.nf = featureSize
        self.nc = channelsDepth
        self.batch_size = batchSize

class HyperParameters:
    def __init__(self, numberGpu, learningRate, beta):
        self.ngpu = numberGpu
        self.lr = learningRate
        self.beta = beta


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


def weights_init(m):     # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def optAdam(netParameters, hyperParams):
    return optim.Adam(netParameters, lr=hyperParams.lr, betas=(hyperParams.beta, 0.999))

def optRMSProp(netParameters, hyperParams):
    return optim.RMSprop(netParameters, lr=hyperParams.lr)

def optSGD(netParameters, hyperParams):
    return optim.SGD(netParameters, lr=hyperParams.lr)

def num_flat_features(x):
    size = x.size()
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class GradientPenalizer :
    def __init__(self, gpWeight, trackProgress=True, useCuda=False) -> None:
        self.useCuda = useCuda
        self.gpWeight = gpWeight
        self.trackProgress = trackProgress
        self.penalties = []
        self.norms = []

    def calculate(self, D, real, fake):
        # Calculate interpolation
        real = torch.reshape(real, (real.size(0), real.size(1)))
        alpha = torch.rand(real.size())
        if self.useCuda:
            alpha = alpha.cuda()
        interpolated = alpha * real.requires_grad_(True) + (1 - alpha) * fake.requires_grad_(True)
        if self.useCuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        ones = torch.ones(prob_interpolated.size())
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                        grad_outputs=ones.cuda() if self.useCuda else ones,
                                        create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height), so flatten to easily take norm per example in batch
        gradients = gradients.view(real.size(0), -1)

        # Derivatives of the gradient close to 0 can cause problems because of the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        if self.trackProgress:
            self.norms.append(gradients.norm(2, dim=1).mean().item())

        # Return gradient petrackProgress=Truenalty
        penalty = self.gpWeight * ((gradients_norm - 1) ** 2).mean()
        if self.trackProgress:
            self.penalties.append(penalty.item())
        return penalty



# DCGAN, use optAdam or optRMSProp optimizers
#
# class Discriminator(nn.Module):
#     def __init__(self, hyper, problem):
#         super(Discriminator, self).__init__()
#         self.ngpu = hyper.ngpu
#         nfx2 = problem.nf * 2
#         nfx4 = problem.nf * 4
#         kernel_size = 3
#         stride = 1
#         self.main = nn.Sequential(
#              # nn.Conv2d(problem.nc, nfx2, kernel_size, stride, 1, bias=False),  # input is 2 x 1 x 1, output of 4x1x1
#              # nn.LeakyReLU(0.2),
#              # nn.Conv2d(nfx2, nfx4, kernel_size, stride, 1, bias=False),  # input is 2 x 1 x 1, output of 4x1x1
#              # nn.LeakyReLU(0.2),
#              # nn.Conv2d(nfx4, 1, kernel_size, stride, 1, bias=False),  # output an answers real/fake
#             nn.Linear(problem.nc, nfx2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(nfx2, 1),
#             nn.Sigmoid()
#         )
#         self.optimizer = optRMSProp(self.parameters(), hyper)
#         # self.optimizer = optAdam(self.parameters(), hyper)
#
#     def forward(self, x):
#         x = torch.reshape(x, (x.size(0), x.size(1)))
#         return self.main(x)
#
#
# class Generator(nn.Module):
#     def __init__(self, hyper, problem):
#         super(Generator, self).__init__()
#         self.ngpu = hyper.ngpu
#         kernel_size = 3
#         stride = 1
#         self.nfx4 = problem.nf * 4
#         self.nfx2 = problem.nf * 2
#         # self.main = nn.Sequential(
#         #     nn.ConvTranspose2d(problem.nz, self.nfx4, 1, stride, 0, bias=False), nn.BatchNorm2d(self.nfx4), nn.ReLU(),
#         #     nn.ConvTranspose2d(self.nfx4, self.nfx2, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(self.nfx2), nn.ReLU(),
#         #     nn.ConvTranspose2d(self.nfx2, problem.nf, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(problem.nf), nn.ReLU(),
#         #     nn.ConvTranspose2d(problem.nf, problem.nc, 1, stride, 0, bias=False), nn.Tanh()
#         # )
#         self.main = nn.Sequential(
#             nn.Linear(problem.nz, self.nfx2),
#             nn.BatchNorm1d(self.nfx2), nn.ReLU(),
#             nn.Linear(self.nfx2, problem.nc))
#             #nn.Sigmoid())
#
#     def forward(self, x):
#         x = torch.reshape(x, (x.size(0), x.size(1)))
#         return self.main(x)
