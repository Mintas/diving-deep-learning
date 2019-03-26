import torch

class Trainer(object):
    def __init__(self, device, problemSize, criterion, initOptimizer, preprocessData):
        self.G_losses = []
        self.D_losses = []
        self.initOptimizer = initOptimizer
        self.device = device
        self.criterion = criterion
        self.realLabels = torch.full((problemSize.batch_size,), 1, device=device)
        self.fakeLabels = torch.full((problemSize.batch_size,), 0, device=device)
        self.noise = lambda: torch.randn(problemSize.batch_size, problemSize.nz, 1, 1, device=device)
        self.prepare = lambda data: preprocessData(data).view(problemSize.batch_size, problemSize.nz, 1, 1).to(device)

    def fwdBwdError(self, net, input, labels):
        # Forward pass real batch through D
        output = net(input).view(-1)
        # Calculate loss on all-real batch
        err = self.criterion(output, labels)
        # Calculate gradients for D in backward pass
        err.backward()
        return err, output.mean().item()

# num_epochs, dataLoader, batch_size, nc, nz, D, G
    def train(self, Dis, Gen, dataLoader, num_epochs, hyperParams, painter=None):
        fixed_noise = self.noise()
        Dis_optimizer = self.initOptimizer(Dis.parameters(), hyperParams)
        Gen_optimizer = self.initOptimizer(Gen.parameters(), hyperParams)
        iters = 0
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataLoader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ## Train with all-real batch
                # Format batch
                real = self.prepare(data)
                Dis.zero_grad()
                errD_real, D_x = self.fwdBwdError(Dis, real, self.realLabels)

                fake = Gen(self.noise())
                errD_fake, D_G_z1 = self.fwdBwdError(Dis, fake.detach(), self.fakeLabels)

                errD = errD_real + errD_fake
                # Update D
                Dis_optimizer.step()
                self.D_losses.append(errD.item())

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                if i % 2 == 0:
                    Gen.zero_grad()
                    errG, D_G_z2 = self.fwdBwdError(Dis, fake, self.realLabels)
                    # Update G
                    Gen_optimizer.step()
                # hint ?
                self.G_losses.append(errG.item())

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataLoader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Check how the generator is doing by saving G's output on fixed_noise
                if painter and (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataLoader) - 1)):
                    with torch.no_grad():
                        fake = Gen(fixed_noise).detach().cpu()
                    painter.plot(fake, data, epoch, iters)

                iters += 1