class ProblemSize:
    def __init__(self, latentInputLength, featureSize, channelsDepth,  batchSize, imgSize):
        self.nz = latentInputLength
        self.nf = featureSize
        self.nc = channelsDepth
        self.batch_size = batchSize
        self.imgSize = imgSize


class HyperParameters:
    def __init__(self, numberGpu, learningRate, beta):
        self.ngpu = numberGpu
        self.lr = learningRate
        self.beta = beta