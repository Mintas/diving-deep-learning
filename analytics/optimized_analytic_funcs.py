import numpy as np
from os.path import dirname

import analytics.plotAnalytics
from analytics.plotAnalytics import doPlotAssymetry, doPlotShowerWidth, doPlotSparsity, plotMeanWithTitle, \
    plotMeanAbsDiff, plotResponses, plotEnergies, comonHistRange, plotResponseImg, doPlotClShape
from analytics import analytic_funcs as AF
from scipy.interpolate import interp2d
##include math
import domain.ecaldata as ED
import plots.plotUi as PUI
import matplotlib.pyplot as plt
import matplotlib
#import serialization.iogan as io

#Below we define set of statistic accumulators, required methods are append(customargs) and get(flag)

class AssymetryAccumulator :
    def __init__(self, imgSize) -> None:
        self.imgSize = imgSize
        self.assymOrtho = []
        self.assymNonOrtho = []
        bound = imgSize/2.0 - 0.5
        x = np.linspace(-bound, bound, imgSize)
        y = np.linspace(-bound, bound, imgSize)
        self.xx, self.yy = np.meshgrid(x, y)

    def append(self, momentum, img, lineOrt, lineNotOrt, sumImg):
        self.assymOrtho.append(AF.doComputeAssym(img, lineOrt, momentum, True, self.xx, self.yy, sumImg))
        self.assymNonOrtho.append(AF.doComputeAssym(img, lineNotOrt, momentum, True, self.xx, self.yy, sumImg))

    def get(self, ortho=True):
        return self.assymOrtho if ortho else self.assymNonOrtho


class ShowerWidthAccumulator :
    def __init__(self, imgSize) -> None:
        self.imgSize = imgSize
        self.widthOrtho = []
        self.widthNonOrtho = []
        bound = imgSize/2.0 - 0.5
        self.x = np.linspace(-bound, bound, imgSize)
        self.y = np.linspace(-bound, bound, imgSize)
        self.x_ = np.linspace(-bound, bound, 100)

    def interpolation(self, img):
        return interp2d(self.x, self.y, img, kind='cubic')

    def append(self, momentum, img, lineOrt, lineNotOrt):
        bb = self.interpolation(img)

        y_Ort = lineOrt(self.x_)
        y_NotOrt = lineNotOrt(self.x_)

        pyFracPx = momentum[1] / momentum[0]
        rescale = np.sqrt(1 + pyFracPx * pyFracPx)
        rescaledX = rescale * self.x_

        self.widthOrtho.append(AF.computeWidth(bb, rescaledX, self.x_, y_Ort))
        self.widthNonOrtho.append(AF.computeWidth(bb, rescaledX, self.x_, y_NotOrt))

    def get(self, ortho=True):
        return self.widthOrtho if ortho else self.widthNonOrtho


class SparsityAccumulator :
    def __init__(self) -> None:
        self.alpha = np.linspace(-5, 0, 50)
        self.sparsity = []

    def append(self, momentum, img, sumImg):
        v_r = []
        for a in self.alpha:
            v_r.append(AF.computeMsRatio2(pow(10, a), img, sumImg))
        self.sparsity.append(v_r)

    def get(self, sparsityOrAlpha=True):
        return self.sparsity if sparsityOrAlpha else self.alpha


class EnergyResponseAccumulator :
    def __init__(self) -> None:
        self.energies = []

    def append(self, momentum, sumImg):
        self.energies.append(sumImg)

    def get(self, ignored=True):
        return self.energies

class ClusterShapeAccumulator : #this one accumulates center-wise energy deposits with borders of width 2
    def __init__(self, imgSize) -> None:
        self.imgSize = imgSize
        self.energies = {}
        for crop in range(min(16, imgSize), min(1, imgSize), -2):
            self.energies[crop] = []

    def append(self, img, sumImg):
        prevImg, prevCrop, prevN, prevAcc = img, 0, 0, [] #todo : divide or not? here we also can divide by sumImg to compute relative statistics
        for crop,nrg in self.energies.items():
            curCropImg = AF.cropCenter(prevImg, crop, crop)
            curN = np.sum(curCropImg)
            if (prevCrop > 0):
                prevAcc.append(prevN - curN)
            if (crop == 2): #the expected minimal center crop
                nrg.append(curN)
            else:
                prevImg, prevCrop, prevN, prevAcc = curCropImg, crop, curN, nrg

    def get(self, ignored=True):
        return self.energies

class AccumEnum :
    ASSYMETRY = "A"
    WIDTH = "W"
    SPARSITY = "S"
    ENERGY = "ER"
    CLUSTER_SHAPE = "CS"

class EcalStats :
    def __init__(self, statsDict) -> None:
        self.stats = statsDict

    # flag represent additional feature of accumulator, True or ignored by default
    def get(self, statType, flag=True):
        return self.stats.get(statType).get(flag)

def optimized_analityc(ecalData, imgsize) :
    response = ecalData.response
    momentum = ecalData.momentum
    points = ecalData.point

    assym = AssymetryAccumulator(imgsize)
    width = ShowerWidthAccumulator(imgsize)
    sprsity = SparsityAccumulator()
    nrgy = EnergyResponseAccumulator()
    clusterShape = ClusterShapeAccumulator(imgsize)

    for i in range(len(response)) :
        img = response[i]
        p = momentum[i]
        point = points[i]

        sumImg = np.sum(img)

        x0, y0 = AF.rotate(p, point)

        lfOrthog = AF.doLineFunc(True, p, x0, y0)
        lfNotOrthog = AF.doLineFunc(False, p, x0, y0)

        nrgy.append(p, sumImg)
        clusterShape.append(img, sumImg)
        assym.append(p, img, lfOrthog, lfNotOrthog, sumImg)
        if i <= 10000 :
            width.append(p, img, lfOrthog, lfNotOrthog)
        if i <= 3000 :
            sprsity.append(p, img, sumImg)

    return EcalStats({AccumEnum.ASSYMETRY : assym,
            AccumEnum.WIDTH : width,
            AccumEnum.SPARSITY : sprsity,
            AccumEnum.ENERGY : nrgy,
            AccumEnum.CLUSTER_SHAPE : clusterShape})


class Layouts :
    DISCOVER = 210
    GENERATED = 220


def runPlotMeans(ecalData, fakeData, haveFake, plotUi):
    if haveFake:
        ecalMean, fakeMean = np.mean(ecalData.response, axis=0, keepdims=False), np.mean(fakeData.response, axis=0,
                                                                                         keepdims=False)
        commonRange, xPostfix = comonHistRange(ecalMean, fakeMean, False)

        print(fakeData.title)

        def plotMeans():
            plt.suptitle('ECAL: mean response of deposited energy', fontsize=16)

            plt.subplot(Layouts.GENERATED + 1)
            plt.title(ecalData.title)
            plotResponseImg(ecalMean, vmin=commonRange[0], vmax=commonRange[1], removeTicks=False)

            plt.subplot(Layouts.GENERATED + 2)
            plt.title(fakeData.title)
            plotResponseImg(fakeMean, vmin=commonRange[0], vmax=commonRange[1], removeTicks=False)

            if (ecalData.response.shape[0] == fakeData.response.shape[0]):
                plt.subplot(Layouts.GENERATED + 3)
                plotResponseImg(abs(ecalMean - fakeMean), vmin=commonRange[0], vmax=commonRange[1], removeTicks=False)

        plotUi.toView(plotMeans)
    else:
        plotUi.toView(lambda: plotMeanWithTitle(ecalData.response, ecalData.title))


def runAnalytics(filename, ecalData, fakeData=None, ecalStats=None, fakeStats=None):
    #print(ecalData.title)
    matplotlib.rcParams.update({'font.size': 13})


    haveFake = fakeData is not None
    outputfile = dirname(dirname(__file__)) + filename + '_stats' + ('_generated' if haveFake else '')
    plotUi = PUI.PDFPlotUi(outputfile)  # PUI.ShowPlotUi()

    imgSize = 30
    #imgSize = ecalData.response[0].shape[0]
    #runPlotMeans(ecalData, fakeData, haveFake, plotUi)

    if ecalStats is None:
        ecalStats = optimized_analityc(ecalData, imgSize)
    if haveFake and fakeStats is None:
        fakeStats = optimized_analityc(fakeData, imgSize)

    #plotUi.toView(lambda: plotResponses(ecalData, True, fakeData))
    #plotUi.toView(lambda: plotResponses(ecalData, False, fakeData))

    layout = Layouts.GENERATED if haveFake else Layouts.DISCOVER

    def runSubplot(func, statType, pos, orth, range=True):
        pos = pos + 1
        plt.subplot(pos)
        func(ecalStats.get(statType, orth), orth,
                        fakeStats.get(statType, orth) if haveFake else None, range)
        return pos

    def runPlotLayoutSublots(type, func):
        position = layout
        position = runSubplot(func, type, position, False)
        if haveFake : position = runSubplot(func, type, position, False, range=False)
        position = runSubplot(func, type, position, True)
        if haveFake : position = runSubplot(func, type, position, True, range=False)

    plotUi.toView(lambda: runPlotLayoutSublots(AccumEnum.ASSYMETRY, doPlotAssymetry))
    plotUi.toView(lambda: runPlotLayoutSublots(AccumEnum.WIDTH, doPlotShowerWidth))


    es = ecalStats.get(AccumEnum.SPARSITY, True)
    plotUi.toView(lambda: doPlotSparsity(es, ecalStats.get(AccumEnum.SPARSITY, False), fakeStats.get(AccumEnum.SPARSITY, True) if haveFake else None))

    plotUi.toView(lambda: runPlotLayoutSublots(AccumEnum.ENERGY, lambda e, o, f, r: plotEnergies(e, o, f, r, imgSize)))

    def doPlotCS(ecal, logScaled, fake, allPoints, cropSize):
        plt.suptitle('Cluster Shape E_c/E_total for c={}x{} '.format(cropSize, cropSize), fontsize=16)
        doPlotClShape(ecalStats.get(AccumEnum.ENERGY), ecal, logScaled,
                        fakeStats.get(AccumEnum.ENERGY) if haveFake else None,
                        fake if haveFake else None, allPoints)

    def runPlotClusterShapeStatistics(crop):
        runPlotLayoutSublots(AccumEnum.CLUSTER_SHAPE,
                             lambda e, o, f, r: doPlotCS(e.get(crop), o, f.get(crop) if haveFake else None, r, crop))
        plotUi.show()
        plotUi.figure()
        runPlotLayoutSublots(AccumEnum.CLUSTER_SHAPE,
                             lambda e, o, f, r: plotEnergies(e.get(crop), o, f.get(crop) if haveFake else None, r, crop))

    for crop in ecalStats.get(AccumEnum.CLUSTER_SHAPE, True):
        plotUi.toView(lambda: runPlotClusterShapeStatistics(crop))

    plotUi.close()
    return ecalStats, fakeStats


import architectures.dcganBatchNorm as dcgan
import domain.parameters
import mygan

def run():
    import torch
    #runAnalytics('caloGAN_v3_case5_2K')
    dataset = 'caloGAN_v3_case4_2K'
    predefinedRanges={AccumEnum.ASSYMETRY : {True: [-0.8, -0.5], False: [-1.0, 0.6]},
            AccumEnum.WIDTH : {True: [2.5, 7.0], False: [2.0, 7.5]},
                      AccumEnum.ENERGY : {True: False, False: False}} #both False here, because we have logScaled as first boolean argument and rangeByExpectedOnly as second boolean

    problem = domain.parameters.ProblemSize(100, 100, 1, 1000, 30)
    hyperParams = domain.parameters.HyperParameters(0, 0.0003, 0.5)

    G = dcgan.GenEcal(mygan.GANS.ECRAMER, hyperParams, problem)
    D = dcgan.DiscEcal(mygan.GANS.ECRAMER, hyperParams, problem)
    #io.loadGANs(D, G, '/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_stats.pth')

    es = torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_stats.pth')
             #torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_dcganZMatch_nf64_stats.pth')
    runAnalytics('/' + 'caloGAN_v4_case2_50K_dcganZMatch', [], [], es, fs)
#run()