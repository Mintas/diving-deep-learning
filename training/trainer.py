import torch
import os
from serialization import iogan


class Debugg(object):
    def __init__(self, print, plot, save) -> None:
        self.print = print
        self.plot = plot
        self.save = save



class Trainer(object):
    def __init__(self, device, problemSize, ganLossCalculator, initOptimizer, preprocessData, path='', debugg=Debugg(5, 25, 1)):
        self.G_losses = []
        self.D_losses = []
        self.initOptimizer = initOptimizer
        self.ganLoss = ganLossCalculator
        self.problemSize = problemSize
        self.device = device
        self.path = path
        self.debugg = debugg
        self.prepare = lambda data: preprocessData(data)

    def noise(self, noiseCount=None):
        count = noiseCount if noiseCount is not None else self.problemSize.batch_size
        return torch.randn(count, self.problemSize.nz, device=self.device)

    def tryLoadPrecomputed(self, Dis, Gen, Dopt, Gopt):
        if not iogan.isFilePresent(self.path): return 0, None
        e, self.D_losses, self.G_losses, fixedNoise = iogan.loadGAN(Dis, Dopt, Gen, Gopt, self.path)
        print('loaded precomputed gans')
        return e, fixedNoise


    def train(self, Dis, Gen, dataLoader, num_epochs, hyperParams, painter=None, fixedNoiseCount=None):
        Dis_optimizer = self.initOptimizer(Dis.parameters(), hyperParams)
        Gen_optimizer = self.initOptimizer(Gen.parameters(), hyperParams)
        computedEpochs, savedNoise = self.tryLoadPrecomputed(Dis, Gen, Dis_optimizer, Gen_optimizer)

        fixed_noise = savedNoise if savedNoise is not None else [self.noise(fixedNoiseCount)]
        datasetLength = len(dataLoader)

        iters = 0
        for epoch in range(computedEpochs, num_epochs + computedEpochs + 1):
            for i, data in enumerate(dataLoader, 0):

                #real = [data[0].to(self.device), data[1].to(self.device)]
                real = [d.to(self.device) for d in data]
                if not savedNoise :
                    fixed_noise.append(real[1][:fixedNoiseCount:100]) # take every 100th sample

                fake = [Gen([self.noise(), real[1]]) for i in range(0, self.ganLoss.needFakes)] #real[1] stands for condition
                D_G_z1, D_x, errD = self.trainDiscriminator(Dis, Dis_optimizer, real, fake)
                self.D_losses.append(errD.item())

                #if i % 2 == 0: # extract to hyperparams as DiscriminatorPerGeneratorTrains
                D_G_z2, errG = self.trainGenerator(Dis, Gen, Gen_optimizer, real, fake)
                self.G_losses.append(errG.item()) # when must we save G_losses

                if i % self.debugg.print == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, datasetLength,
                             errD.item(), errG.item(), D_x.mean().item(), D_G_z1.mean().item(), D_G_z2.mean().item()))

                # Check how the generator is doing by saving G's output on fixed_noise
                if painter and (iters % self.debugg.plot == 0) or ((epoch == num_epochs + computedEpochs) and (i == datasetLength - 1)):
                    with torch.no_grad():
                        fake = Gen(fixed_noise).detach().cpu()
                    painter.plot(fake, data, epoch, iters)

                iters += 1
            if epoch % self.debugg.save == 0:
                iogan.saveGAN(epoch, fixed_noise, Dis, Dis_optimizer, self.D_losses, Gen, Gen_optimizer, self.G_losses, self.path)

        return Dis_optimizer, Gen_optimizer

    def trainDiscriminator(self, Dis, Dis_optimizer, data, fake): #data is [image, condition], fake is output image for conditions
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        Dis.zero_grad()
        D_G_z1, D_x, errD = self.ganLoss.forwardBackwardD(Dis, data, fake)
        # Update D
        Dis_optimizer.step()
        return D_G_z1, D_x, errD

    def trainGenerator(self, Dis, Gen, Gen_optimizer, data, fake):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        Gen.zero_grad()
        D_G_z2, errG  = self.ganLoss.forwardBackwardG(Dis, data, fake)
        # Update G
        Gen_optimizer.step()
        return D_G_z2, errG