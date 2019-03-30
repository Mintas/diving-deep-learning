import torch

class GanLoss(object):
    def __init__(self, device, problemSize, criterion) -> None:
        self.criterion = criterion
        self.realLabels = torch.full((problemSize.batch_size,), 1, device=device)
        self.fakeLabels = torch.full((problemSize.batch_size,), 0, device=device)

    def fwdBwdError(self, D, input, labels):
        output = D(input).view(-1) # Forward pass through D
        loss = self.criterion(output, labels)
        loss.backward()  # Calculate gradients
        return output, loss

    def forwardBackwardG(self, D, fake):
        return self.fwdBwdError(D, fake, self.realLabels)

    def forwardBackwardD(self, D, real, fake):
        D_x, errD_real  = self.fwdBwdError(D, real, self.realLabels)

        D_G_z1, errD_fake = self.fwdBwdError(D, fake.detach(), self.fakeLabels)
        errD = errD_real + errD_fake
        return D_G_z1, D_x, errD

class WganLoss(object):
    def __init__(self, problemSize, gradientPenalizer) -> None:
        self.gradientPenalizer = gradientPenalizer

    def forwardBackwardG(self, D, fake):
        D_G_z2 = D(fake)
        errG = - D_G_z2.mean()
        errG.backward()
        return D_G_z2, errG

    def forwardBackwardD(self, D, real, fake):
        D_x = D(real)
        fake = fake.detach()
        D_G_z1 = D(fake)

        # Get gradient penalty
        gradient_penalty = self.gradientPenalizer.calculate(D, real, fake)
        # self.losses['GP'].append(gradient_penalty.data[0])

        # Create total loss and optimize
        errD = D_G_z1.mean() - D_x.mean() + gradient_penalty
        errD.backward()
        return D_G_z1, D_x, errD



class Trainer(object):
    def __init__(self, device, problemSize, ganLossCalculator, initOptimizer, preprocessData):
        self.G_losses = []
        self.D_losses = []
        self.initOptimizer = initOptimizer
        self.ganLoss = ganLossCalculator
        self.noise = lambda: torch.randn(problemSize.batch_size, problemSize.nz, 1, 1, device=device)
        self.prepare = lambda data: preprocessData(data).view(problemSize.batch_size, problemSize.nz, 1, 1).to(device)

    def train(self, Dis, Gen, dataLoader, num_epochs, hyperParams, painter=None):
        fixed_noise = self.noise()
        Dis_optimizer = self.initOptimizer(Dis.parameters(), hyperParams)
        Gen_optimizer = self.initOptimizer(Gen.parameters(), hyperParams)
        datasetLength = len(dataLoader)

        iters = 0
        for epoch in range(num_epochs):
            for i, data in enumerate(dataLoader, 0):
                fake = Gen(self.noise())
                D_G_z1, D_x, errD = self.trainDiscriminator(Dis, Dis_optimizer, data, fake)
                self.D_losses.append(errD.item())

                if i % 2 == 0: # extract to hyperparams as DiscriminatorPerGeneratorTrains
                    D_G_z2, errG = self.trainGenerator(Dis, Gen, Gen_optimizer, fake)
                self.G_losses.append(errG.item()) # when must we save G_losses

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, datasetLength,
                             errD.item(), errG.item(), D_x.mean().item(), D_G_z1.mean().item(), D_G_z2.mean().item()))

                # Check how the generator is doing by saving G's output on fixed_noise
                if painter and (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == datasetLength - 1)):
                    with torch.no_grad():
                        fake = Gen(fixed_noise).detach().cpu()
                    painter.plot(fake, data, epoch, iters)

                iters += 1

    def trainDiscriminator(self, Dis, Dis_optimizer, data, fake):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        real = self.prepare(data)
        Dis.zero_grad()
        D_G_z1, D_x, errD = self.ganLoss.forwardBackwardD(Dis, real, fake)
        # Update D
        Dis_optimizer.step()
        return D_G_z1, D_x, errD

    def trainGenerator(self, Dis, Gen, Gen_optimizer, fake):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        Gen.zero_grad()
        D_G_z2, errG  = self.ganLoss.forwardBackwardG(Dis, fake)
        # Update G
        Gen_optimizer.step()
        return D_G_z2, errG