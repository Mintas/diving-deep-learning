import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from IPython.display import HTML

def plotLosses(G_losses, D_losses):
    plt.figure()
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
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
        binWidth = (ranges[1] - ranges[0]) / self.maxBins
        bins = np.arange(ranges[0], ranges[1] + binWidth, binWidth)

        _, ax = plt.subplots()
        plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
        ax.hist(fake, bins=bins, alpha=1, density=True, label='fake generated')
        ax.hist(data, bins=bins, alpha=0.75, density=True, label='data sampled')
        if self.curveSample :
            plt.plot(self.curveSample[0], self.curveSample[1])
        plt.show()

#  for 2D arrays, curves
class CurvePainter():
    def plotHist(self, fake, data, epoch, iters):
        reshape = fake.reshape((fake.size(0), fake.size(1)))

        sorted = reshape[reshape[:, 0].argsort()]
        reshaped = torch.t(sorted)
        asList = reshaped.tolist()
        plt.title("Generator: epoch " + str(epoch) + " and iteration " + str(iters))
        plt.plot(asList[0], asList[1], label="FromNoise")
        plt.plot(np.sort(data[0]), data[1][np.argsort(data[0])].numpy(), label="sampledBatch")
        plt.show()