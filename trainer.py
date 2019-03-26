import torch

class Trainer(object):
    def __init__(self, device, problemSize, criterion, initOptimizer, preprocessData):
        self.G_losses = []
        self.D_losses = []
        self.initOptimizer = initOptimizer
        self.criterion = criterion
        self.realLabels = torch.full((problemSize.batch_size,), 1, device=device)
        self.fakeLabels = torch.full((problemSize.batch_size,), 0, device=device)
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
                fakeGenerated = Gen(self.noise())
                D_G_z1, D_x, errD = self.trainDiscriminator(Dis, Dis_optimizer, data, fakeGenerated)
                self.D_losses.append(errD.item())

                if i % 2 == 0: # extract to hyperparams as DiscriminatorPerGeneratorTrains
                    D_G_z2, errG = self.trainGenerator(Dis, Gen, Gen_optimizer, fakeGenerated)
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
        errD_real, D_x = self.fwdBwdError(Dis, real, self.realLabels)

        errD_fake, D_G_z1 = self.fwdBwdError(Dis, fake.detach(), self.fakeLabels)
        errD = errD_real + errD_fake
        # Update D
        Dis_optimizer.step()
        return D_G_z1, D_x, errD

    def trainGenerator(self, Dis, Gen, Gen_optimizer, fakeGenerated):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        Gen.zero_grad()
        errG, D_G_z2 = self.fwdBwdError(Dis, fakeGenerated, self.realLabels)
        # Update G
        Gen_optimizer.step()
        return D_G_z2, errG

    def fwdBwdError(self, net, input, labels):
        output = net(input).view(-1) # Forward pass through D
        loss = self.criterion(output, labels)
        loss.backward()  # Calculate gradients
        return loss, output