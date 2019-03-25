import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous as rvc


class SplicedNormCurve:
    def __init__(self, mean1, mean2, intervalWidth):
        self.normal1 = Normal(mean1, 1)
        self.normal2 = Normal(mean2, 1)
        self.splicePoint = (mean1 + mean2) / 2.0
        self.normConstant = 2 * self.normal1.cdf(self.splicePoint)
        self.interval = torch.distributions.Uniform(mean1 - intervalWidth, mean2 + intervalWidth)
        self.vectorizeCurve = np.vectorize(self.point)

    def curve(self, x):
        return self.prob(x, self.normal1 if x < self.splicePoint else self.normal2)

    def prob(self, x, distr):
        return torch.exp(distr.log_prob(x)) / self.normConstant

    def point(self, x):
        return x, self.curve(x)

    def sampleCurve(self, sampleSize):
        sortedArguments = torch.sort(self.interval.sample((sampleSize,)))
        return self.vectorizeCurve(sortedArguments[0])

class FromPdfDistribution(rvc):
    def __init__(self, pdfunction, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None,
                 extradoc=None, seed=None):
        super().__init__(momtype, a, b, xtol, badvalue, name, longname, shapes, extradoc, seed)
        self.pdfunction = pdfunction

    def _pdf(self, x, *args):
        return self.pdfunction(x)


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

class ProbDistrDataset(tdata.Dataset):
    def __init__(self, distr, len) -> None:
        super().__init__()
        self.len = len
        self.distr = distr
        self.xPoints = distr.sample((len,))

    def __getitem__(self, index):
        return self.xPoints[index]

    def __len__(self):
        return self.len

from scipy import integrate

def run():
    m1 = 0
    m2 = 2
    sampleSize = 1280
    spread = 3

    curve = SplicedNormCurve(m1, m2, spread)
    points = curve.sampleCurve(sampleSize)

    plt.plot(points[0], points[1])
    plt.show()