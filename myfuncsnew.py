import torch
from torch.distributions.normal import Normal
import numpy as np
import myfuncs
import matplotlib.pyplot as plt
from scipy.optimize import root
from numpy.random import uniform

class SplicedNormCurveMany:
    def __init__(self, means, intervalWidth):
        self.cnt = len(means)
        self.normals = [Normal(mean, 1) for mean in means]
        self.splicePoints = [(means[i] + means[i+1]) / 2 for i in range(0, self.cnt-1)]
        self.leftCdf = [self.normals[i].cdf(sp) for (i,sp) in enumerate(self.splicePoints)] # is equal to cdf1(splicePoint) / C
        self.rightCdf = [self.normals[i+1].cdf(sp) for (i, sp) in enumerate(self.splicePoints)]
        self.cumNorms = self.cumNomrs()
        self.normConstant = self.cumNorms[self.cnt-1]
        self.interval = torch.distributions.Uniform(means[0] - intervalWidth, means[self.cnt-1] + intervalWidth)
        self.vectorizeCurve = np.vectorize(self.point)
        self.vectorizeCDF = np.vectorize(lambda x: (x, self.cdfCurve(x)))

    def cumNomrs(self):
        cumNrm = [self.leftCdf[0]]
        for i in range(1, self.cnt-1) :
            cumNrm.append(cumNrm[i-1] + self.leftCdf[i] - self.rightCdf[i-1])
        cumNrm.append(cumNrm[self.cnt-2] + (1 - self.rightCdf[self.cnt-2]))
        return cumNrm

    def curve(self, x):
        # piecewise = np.piecewise(x, [x < self.splicePoint, x >= self.splicePoint], [self.normal1.log_prob, self.normal2.log_prob])
        # return torch.exp(piecewise / self.normConstant)
        with torch.no_grad():
            for i in range(0, self.cnt-1) :
                if x < self.splicePoints[i] :
                    return self.prob(x, self.normals[i])
            return self.prob(x, self.normals[self.cnt-1])

    def cdfCurve(self, x):
        with torch.no_grad():
            if x < self.splicePoints[0] :
                return self.normed(self.normals[0].cdf(x))
            for i in range(1, self.cnt-1) :
                if x < self.splicePoints[i] :
                    return self.normed(self.normals[i].cdf(x) - self.rightCdf[i-1] + self.cumNorms[i-1])
            return self.normed(self.normals[self.cnt-1].cdf(x) - self.rightCdf[self.cnt-2] + self.cumNorms[self.cnt-2])

    def prob(self, x, distr):
        return self.normed(torch.exp(distr.log_prob(x)))

    def normed(self, x):
        return x / self.normConstant

    def point(self, x):
        return x, self.curve(x)

    def sample(self, sampleSize):
        return myfuncs.sampleCdf(self.cdfCurve, sampleSize) # mb it will be nice to give splicePoint as initial guess

    def sampleCurve(self, sampleSize):
        sortedArguments = torch.sort(self.interval.sample((sampleSize,)))
        return self.vectorizeCurve(sortedArguments[0])

    def sampleCDF(self, sampleSize):
        sortedArguments = torch.sort(self.interval.sample((sampleSize,)))
        return self.vectorizeCDF(sortedArguments[0])
