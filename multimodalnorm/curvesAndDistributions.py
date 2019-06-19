import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.random import uniform

def sampleCdf(cdf, num_samples, guess=1/2) :
    seeds = uniform(0, 1, num_samples)
    samples = []
    for seed in seeds:
        shifted = lambda x: cdf(x[0])-seed
        soln = root(shifted, guess)
        samples.append(np.float32(soln.x[0]))
    return np.array(samples)


class SplicedNormCurve:
    def __init__(self, mean1, mean2, intervalWidth):
        self.normal1 = Normal(mean1, 1)
        self.normal2 = Normal(mean2, 1)
        self.splicePoint = (mean1 + mean2) / 2.0
        self.addCdf1 = 1/2 # is equal to cdf1(splicePoint) / C
        self.normConstant = 2 * self.normal1.cdf(self.splicePoint)
        self.substractCdf2 = self.normal2.cdf(self.splicePoint) / self.normConstant
        self.interval = torch.distributions.Uniform(mean1 - intervalWidth, mean2 + intervalWidth)
        self.vectorizeCurve = np.vectorize(self.point)
        self.vectorizeCDF = np.vectorize(lambda x: (x, self.cdfCurve(x)))

    def curve(self, x):
        # piecewise = np.piecewise(x, [x < self.splicePoint, x >= self.splicePoint], [self.normal1.log_prob, self.normal2.log_prob])
        # return torch.exp(piecewise / self.normConstant)
        with torch.no_grad():
            return self.prob(x, self.normal1 if x < self.splicePoint else self.normal2)

    def cdfCurve(self, x):
        with torch.no_grad():
            return (self.normal1.cdf(x) / self.normConstant) if x < self.splicePoint else ((self.normal2.cdf(x) / self.normConstant) + self.addCdf1 - self.substractCdf2)

    def prob(self, x, distr):
        return torch.exp(distr.log_prob(x)) / self.normConstant

    def point(self, x):
        return x, self.curve(x)

    def sample(self, sampleSize):
        return sampleCdf(self.cdfCurve, sampleSize) # mb it will be nice to give splicePoint as initial guess

    def sampleCurve(self, sampleSize):
        sortedArguments = torch.sort(self.interval.sample((sampleSize,)))
        return self.vectorizeCurve(sortedArguments[0])

    def sampleCDF(self, sampleSize):
        sortedArguments = torch.sort(self.interval.sample((sampleSize,)))
        return self.vectorizeCDF(sortedArguments[0])


import torch.utils.data as tdata
class CurveDataset(tdata.Dataset):
    def __init__(self, len, function, argumentDistr) -> None:
        super().__init__()
        self.curve = function
        self.len = len
        self.argDistr = argumentDistr
        self.xPoints = torch.sort(self.argDistr.sample((len,)))[0]

    def __getitem__(self, index):
        xi = self.xPoints[index]
        return xi, self.curve(xi)

    def __len__(self):
        return self.len

    def preprocess(self, data):
        return torch.from_numpy(np.column_stack((data[0], data[1])))


class ProbDistrDataset(tdata.Dataset):
    def __init__(self, distr, len, preloaded=None) -> None:
        super().__init__()
        self.len = len
        self.distr = distr
        self.xPoints = preloaded if preloaded is not None else distr.sample((len,))

    def __getitem__(self, index):
        return self.xPoints[index]

    def __len__(self):
        return self.len

    def preprocess(self, data):
        return data #torch.reshape(data, (data.size(0), data.size(1)))


def run():
    m1 = 0
    m2 = 5
    sampleSize = 1280
    spread = 3

    curve = SplicedNormCurve(m1, m2, spread)
    plotPdfAndCdf(curve, sampleSize)

    sampled = sampleCdf(curve.cdfCurve, 10)
    print(sampled)


def plotPdfAndCdf(curve, sampleSize):
    points = curve.sampleCurve(sampleSize)
    plt.plot(points[0], points[1])
    points = curve.sampleCDF(sampleSize)
    plt.plot(points[0], points[1])
    plt.show()
