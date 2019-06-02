import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
import analytic_funcs as AF

def plotLosses(G_losses, D_losses):
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plotGradPenalties(penalties, norms):
    plt.figure()
    plt.title("Gradient Penalties and Norms During Training")
    plt.plot(penalties, label="P")
    plt.plot(norms, label="N")
    plt.xlabel("iterations")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# for 1D arrays, distrubitions
class HistorgramPainter():
    def __init__(self, curveSample = None, maxBins = 50):
        self.maxBins = maxBins
        self.curveSample = curveSample

    def plot(self, fake, data, epoch, iters):
        fake = fake.reshape(fake.size(0))
        ranges = [min(min(fake), min(data)), max(max(fake), max(data))]
        bins = self.calcBins(ranges)

        _, ax = plt.subplots()
        plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
        ax.hist(fake, bins=bins, alpha=1, density=True, label='fake generated')
        ax.hist(data, bins=bins, alpha=0.75, density=True, label='data sampled')
        if self.curveSample :
            plt.plot(self.curveSample[0], self.curveSample[1])
        plt.show()

    def plotFake(self, fake, epoch, iters):
        fake = fake.reshape(fake.size(0)).detach()
        ranges = [min(fake), max(fake)]
        bins = self.calcBins(ranges)
        _, ax = plt.subplots()
        plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
        ax.hist(fake, bins=bins, alpha=1, density=True, label='fake generated')
        if self.curveSample :
            plt.plot(self.curveSample[0], self.curveSample[1])
        plt.show()

    def calcBins(self, ranges):
        binWidth = (ranges[1] - ranges[0]) / self.maxBins
        bins = np.arange(ranges[0], ranges[1] + binWidth, binWidth)
        return bins


#  for 2D arrays, curves
class CurvePainter():
    def plot(self, fake, data, epoch, iters):
        self.doPlotFake(epoch, fake, iters)
        plt.plot(np.sort(data[0]), data[1][np.argsort(data[0])].numpy(), label="sampledBatch")
        plt.show()

    def plotFake(self, fake, epoch, iters):
        self.doPlotFake(epoch, fake, iters)
        plt.show()

    def doPlotFake(self, epoch, fake, iters):
        reshape = fake.reshape((fake.size(0), fake.size(1)))
        sorted = reshape[reshape[:, 0].argsort()]
        reshaped = torch.t(sorted)
        asList = reshaped.tolist()
        plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
        plt.plot(asList[0], asList[1], label="FromNoise")

# for ECal responses
class ECalPainter():
    def __init__(self, plotUi):
        self.plotUi = plotUi

    def plot(self, fake, data, epoch, iters):
        self.plotFake(fake, epoch, iters)

    def plotFake(self, fake, epoch, iters):
        plt.suptitle("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
        fake = fake.view(fake.size(0), fake.size(2), fake.size(3))
        self.plotUi.toView(lambda: AF.plotResponses(fake, fake.size(0), [], False))
