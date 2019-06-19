import torch


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
        D_x = torch.chunk(D(real), 2)
        D_x1, D_x2 = D_x[0], D_x[1]
        D_G_z = torch.chunk(D(fake[0]), 2)
        D_G_z1 = D_G_z[1]
        D_G_z2 = D_G_z[0]

        errG = (self.norm(D_x1, D_G_z1) + self.norm(D_x2, D_G_z2) - self.norm(D_G_z1, D_G_z2) - self.norm(D_x1, D_x2)).mean()
        errG.backward()
        return D_G_z2, errG

    def norm(self, x, y):
        return torch.norm(x - y, p=2, dim=-1)

    def forwardBackwardD(self, D, real, fake):
        reals = torch.chunk(real, 2)
        D_x1 = D(reals[0])
        fakes = torch.chunk(fake[0], 2)
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


class GradientPenalizer :
    def __init__(self, gpWeight, trackProgress=True, useCuda=False) -> None:
        self.useCuda = useCuda
        self.gpWeight = gpWeight
        self.trackProgress = trackProgress
        self.penalties = []
        self.norms = []

    def calculate(self, D, real, fake):
        # Calculate interpolation
        # todo : fix for old curves dataset (uncomment while not fixed
        #real = torch.reshape(real, (real.size(0), real.size(1)))
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