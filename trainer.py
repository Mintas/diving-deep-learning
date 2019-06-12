import torch
import os
import iogan

class GanLoss(object):
    def __init__(self, device, problemSize, criterion) -> None:
        self.criterion = criterion
        self.realLabels = torch.full((problemSize.batch_size,), 1, device=device)
        self.fakeLabels = torch.full((problemSize.batch_size,), 0, device=device)
        self.needFakes = 1

    def fwdBwdError(self, D, input, labels):
        output = D(input).view(-1) # Forward pass through D
        loss = self.criterion(output, labels)
        loss.backward()  # Calculate gradients
        return output, loss

    def forwardBackwardG(self, D, real, fake):
        return self.fwdBwdError(D, fake[0], self.realLabels)

    def forwardBackwardD(self, D, real, fake):
        D_x, errD_real  = self.fwdBwdError(D, real, self.realLabels)

        D_G_z1, errD_fake = self.fwdBwdError(D, fake[0].detach(), self.fakeLabels)
        errD = errD_real + errD_fake
        return D_G_z1, D_x, errD

class WganLoss(object):
    def __init__(self, problemSize, gradientPenalizer) -> None:
        self.gradientPenalizer = gradientPenalizer
        self.needFakes = 1

    def forwardBackwardG(self, D, real, fake):
        D_G_z2 = D(fake[0])
        errG = - D_G_z2.mean()
        errG.backward()
        return D_G_z2, errG

    def forwardBackwardD(self, D, real, fake):
        D_x = D(real)
        fake = fake[0].detach()
        D_G_z1 = D(fake)

        # Get gradient penalty
        gradient_penalty = self.gradientPenalizer.calculate(D, real, fake)
        # Create total loss and optimize
        errD = D_G_z1.mean() - D_x.mean() + gradient_penalty
        errD.backward()
        return D_G_z1, D_x, errD


class CramerGanLoss(object):
    def __init__(self, problemSize, gradientPenalizer) -> None:
        self.gradientPenalizer = gradientPenalizer
        self.needFakes = 2

    def forwardBackwardG(self, D, real, fake):
        D_x = D(real)
        D_G_z1 = D(fake[1])
        D_G_z2 = D(fake[0])

        errG = (self.norm(D_x, D_G_z1) + self.norm(D_x, D_G_z2) - self.norm(D_G_z1, D_G_z2)).mean()
        errG.backward()
        return D_G_z2, errG

    def norm(self, x, y):
        return torch.norm(x - y, p=2, dim=-1)

    def forwardBackwardD(self, D, real, fake):
        D_x = D(real)
        fake1 = fake[0].detach()
        D_G_z1 = D(fake1)
        D_G_z2 = D(fake[1].detach())

        # Get gradient penalty; not by D, but as Critic(D, interpolated, fake)
        DCritic = lambda interp : self.critic(D(interp), D_G_z1)
        gradient_penalty = self.gradientPenalizer.calculate(DCritic, real, fake1)
        # Create total loss and optimize
        errD = - torch.mean(self.critic(D_x, D_G_z2) - self.critic(D_G_z1, D_G_z2)) + gradient_penalty
        errD.backward()
        return D_G_z1, D_x, errD

    def critic(self, x, y):
        return self.norm(x, y) - torch.norm(x, p=2, dim=-1)

class CramerEneryGanLoss(object):
    def __init__(self, problemSize, gradientPenalizer) -> None:
        self.gradientPenalizer = gradientPenalizer
        self.needFakes = 1

    def forwardBackwardG(self, D, real, fake):
        D_x = torch.split(D(real), 2)
        D_x1, D_x2 = D_x[0], D_x[1]
        D_G_z = torch.split(D(fake[0]), 2)
        D_G_z1 = D_G_z[1]
        D_G_z2 = D_G_z[0]

        errG = (self.norm(D_x1, D_G_z1) + self.norm(D_x2, D_G_z2) - self.norm(D_G_z1, D_G_z2) - self.norm(D_x1, D_x2)).mean()
        errG.backward()
        return D_G_z2, errG

    def norm(self, x, y):
        return torch.norm(x - y, p=2, dim=-1)

    def forwardBackwardD(self, D, real, fake):
        reals = torch.split(real, 2)
        D_x1 = D(reals[0])
        fakes = torch.split(fake[0], 2)
        fake1 = fakes[0].detach()
        D_G_z1 = D(fake1)
        D_G_z2 = D(fakes[1].detach())

        # Get gradient penalty; not by D, but as Critic(D, interpolated, fake)
        DCritic = lambda interp : self.critic(D(interp), D_G_z1)
        gradient_penalty = self.gradientPenalizer.calculate(DCritic, reals[0], fake1)
        # Create total loss and optimize
        errD = - torch.mean(self.critic(D_x1, D_G_z2) - self.critic(D_G_z1, D_G_z2)) + gradient_penalty
        errD.backward()
        return D_G_z1, D_x1, errD

    def critic(self, x, y):
        return self.norm(x, y) - torch.norm(x, p=2, dim=-1)

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
        self.prepare = lambda data: preprocessData(data).view(problemSize.batch_size, problemSize.nc, problemSize.imgSize, problemSize.imgSize)

    def noise(self, noiseCount=None):
        count = noiseCount if noiseCount is not None else self.problemSize.batch_size
        return torch.randn(count, self.problemSize.nz, 1, 1, device=self.device)

    def tryLoadPrecomputed(self, Dis, Gen, Dopt, Gopt):
        if not os.path.isfile(self.path): return 0, None #'./computed/caloGAN_v3_case1_50K.pdf'
        e, self.D_losses, self.G_losses, fixedNoise = iogan.loadGAN(Dis, Dopt, Gen, Gopt, self.path)
        print('loaded precomputed gans')
        return e, fixedNoise


    def train(self, Dis, Gen, dataLoader, num_epochs, hyperParams, painter=None, fixedNoiseCount=None):
        Dis_optimizer = self.initOptimizer(Dis.parameters(), hyperParams)
        Gen_optimizer = self.initOptimizer(Gen.parameters(), hyperParams)
        computedEpochs, savedNoise = self.tryLoadPrecomputed(Dis, Gen, Dis_optimizer, Gen_optimizer)

        fixed_noise = savedNoise if savedNoise is not None else self.noise(fixedNoiseCount)
        datasetLength = len(dataLoader)

        iters = 0
        for epoch in range(computedEpochs, num_epochs + computedEpochs + 1):
            for i, data in enumerate(dataLoader, 0):
                fake = [Gen(self.noise()) for i in range(0, self.ganLoss.needFakes)]
                real = self.prepare(data).to(self.device)
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

    def trainDiscriminator(self, Dis, Dis_optimizer, data, fake):
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