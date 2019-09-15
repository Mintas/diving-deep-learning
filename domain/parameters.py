class ProblemSize:
    def __init__(self, latentInputLength, featureSize, channelsDepth,  batchSize, imgSize, conditionSize=0):
        self.nz = latentInputLength
        self.nf = featureSize
        self.nc = channelsDepth
        self.batch_size = batchSize
        self.imgSize = imgSize
        self.cs = conditionSize


class HyperParameters:
    def __init__(self, numberGpu, learningRate, beta):
        self.ngpu = numberGpu
        self.lr = learningRate
        self.beta = beta