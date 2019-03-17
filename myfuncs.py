import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt


class JoinNormCurve:
    def __init__(self, mean1, mean2, intervalWidth):
        self.normal1 = Normal(mean1, 1)
        self.normal2 = Normal(mean2, 1)
        self.joinPoint = (mean1 + mean2) / 2.0
        self.interval = torch.distributions.Uniform(mean1 - intervalWidth, mean2 + intervalWidth)
        self.vectorizeCurve = np.vectorize(self.point)

    def curve(self, x):
        if x < self.joinPoint:
            return torch.exp(self.normal1.log_prob(x))
        else:
            return torch.exp(self.normal2.log_prob(x))

    def point(self, x):
        return x, self.curve(x)

    def sampleCurve(self, sampleSize):
        sortedArguments = torch.sort(self.interval.sample((sampleSize,)))
        return self.vectorizeCurve(sortedArguments[0])

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

def run():
    m1 = 0
    m2 = 0
    sampleSize = 128
    spread = 3

    curve = JoinNormCurve(m1, m2, spread)
    points = curve.sampleCurve(sampleSize)

    plt.plot(points[0], points[1])
    plt.show()