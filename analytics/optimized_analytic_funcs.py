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

    p, point, x0, y0, lfOrthog, lfNotOrthog = ([], [], 0, 0, None, None)

    for i in range(len(response)) :
        img = response[i]

        if (i % 100 == 0) :
            p = momentum[i]
            point = points[i]

            sumImg = np.sum(img)

            x0, y0 = AF.rotate(p, point)

            lfOrthog = AF.doLineFunc(True, p, x0, y0)
            lfNotOrthog = AF.doLineFunc(False, p, x0, y0)

            if (i % 10000 == 0) :
                print('stats processed {}'.format(i))

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


def runAnalytics(filename, ecalData, fakeData=None, ecalStats=None, fakeStats=None, plotShape=True):
    print(ecalData.title)
    matplotlib.rcParams.update({'font.size': 13})


    haveFake = fakeData is not None
    outputfile = dirname(dirname(__file__)) + filename + '_stats' + ('_generated' if haveFake else '')
    plotUi = PUI.PDFPlotUi(outputfile)  # PUI.ShowPlotUi()

    #imgSize = 30
    imgSize = ecalData.response[0].shape[0]
    runPlotMeans(ecalData, fakeData, haveFake, plotUi)

    if ecalStats is None:
        ecalStats = optimized_analityc(ecalData, imgSize)
    if haveFake and fakeStats is None:
        fakeStats = optimized_analityc(fakeData, imgSize)

    print('Stats COMPUTED !!!')

    plotUi.toView(lambda: plotResponses(ecalData, True, fakeData))
    plotUi.toView(lambda: plotResponses(ecalData, False, fakeData))

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
    print('ASSYMETRY plotted !!!')

    plotUi.toView(lambda: runPlotLayoutSublots(AccumEnum.WIDTH, doPlotShowerWidth))
    print('WIDTH plotted !!!')


    es = ecalStats.get(AccumEnum.SPARSITY, True)
    plotUi.toView(lambda: doPlotSparsity(es, ecalStats.get(AccumEnum.SPARSITY, False), fakeStats.get(AccumEnum.SPARSITY, True) if haveFake else None))
    print('SPARSITY plotted !!!')

    plotUi.toView(lambda: runPlotLayoutSublots(AccumEnum.ENERGY, lambda e, o, f, r: plotEnergies(e, o, f, r, imgSize)))
    print('ENERGY plotted !!!')

    def doPlotCS(ecal, logScaled, fake, allPoints, cropSize):
        plt.suptitle('Cluster Shape E_c/E_total for c={}x{} '.format(cropSize, cropSize), fontsize=16)
        doPlotClShape(ecalStats.get(AccumEnum.ENERGY), ecal, logScaled,
                        fakeStats.get(AccumEnum.ENERGY) if haveFake else None,
                        fake if haveFake else None, allPoints)

    if plotShape :
        print('Gonna plot SHAPE !!!')
        def runPlotClusterShapeStatistics(crop):
            runPlotLayoutSublots(AccumEnum.CLUSTER_SHAPE,
                             lambda e, o, f, r: doPlotCS(e.get(crop), o, f.get(crop) if haveFake else None, r, crop))
            plotUi.show()
            plotUi.figure()
            runPlotLayoutSublots(AccumEnum.CLUSTER_SHAPE,
                             lambda e, o, f, r: plotEnergies(e.get(crop), o, f.get(crop) if haveFake else None, r, crop))

        for crop in ecalStats.get(AccumEnum.CLUSTER_SHAPE, True):
            plotUi.toView(lambda: runPlotClusterShapeStatistics(crop))
            print('SHAPE plotted crop = {} !!!'.format(crop))

    print('PLOTUI is gonna get closed now!!!')

    plotUi.close()
    return ecalStats, fakeStats



import domain.parameters
import mygan

def run():
    import torch
    #runAnalytics('caloGAN_v3_case5_2K')
    dataset = 'caloGAN_v3_case4_2K'
    #es, fs = runAnalytics('/' + dataset, ecalData = ED.parseEcalData(dataset), fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))

    problem = domain.parameters.ProblemSize(100, 100, 1, 1000, 30)
    hyperParams = domain.parameters.HyperParameters(0, 0.0003, 0.5)

    #import architectures.dcganBatchNorm as dcgan
    #G = dcgan.GenEcal(mygan.GANS.ECRAMER, hyperParams, problem)
    #D = dcgan.DiscEcal(mygan.GANS.ECRAMER, hyperParams, problem)
    #io.loadGANs(D, G, '/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_stats.pth')

    es,fs = torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_stats.pth'),\
            torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/caloGAN_v4_case2_50K_dcganZMatch_nf64_stats.pth')
    runAnalytics('/' + 'caloGAN_v4_case2_50K_dcganZMatch', [], [], es, fs)
#run()
#############################################
#############################################
#############################################
#############################################

def conf():
    import architectures.linearGan21062019 as lineargan
    from analytics.plotAnalytics import doPlotAssymetryArr, doPlotShowerWidthArr, plotEnergiesArr
    import torch
    import os

    basepath = '/Users/mintas/PycharmProjects/untitled1/resources/computed/test/'

    problem = domain.parameters.ProblemSize(100, 120, 1, 1000, 30)
    hyperParams = domain.parameters.HyperParameters(0, 0.0003, 0.5)

    #load networks
    def getStats(datasetName):
        statsname = basepath + 'gen_stats_%s.pth' % datasetName
        generatedName = basepath + 'gen_resp_%s.pth' % datasetName
        estats = torch.load(basepath + '%s_stats.pth' % datasetName, map_location='cpu')
        ecalData = np.load('/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' % datasetName)
        ecalDataPojo = domain.ecaldata.dictToEcalData(ecalData)
        shape = ecalDataPojo.response.shape

        if not os.path.isfile(statsname):
            G = lineargan.GenEcal(mygan.GANS.ECRAMER, hyperParams, problem)
            checkpoint = torch.load('/Users/mintas/PycharmProjects/untitled1/resources/computed/test/swapdataset_%s_linear.pth' % datasetName, map_location='cpu')
            G.load_state_dict(checkpoint['G_state'])


            with torch.no_grad():
                G.eval()
                tonnsOfNoise = torch.randn(shape[0], 100, 1, 1)
                generated = G(tonnsOfNoise)
                torch.save(generated, generatedName)
                print('saved generated responses')
                # generated = generated*generated #comment this to remove sqr after sqrt learning
                # generated = torch.expm1(generated) #comment this to remove sqr after sqrt learning
                fakeData = domain.ecaldata.EcalData(generated.reshape(shape).cpu().numpy(), ecalDataPojo.momentum, ecalDataPojo.point)
            fakeStats = optimized_analityc(fakeData, 30)
            torch.save(fakeStats, statsname)
            return estats, ecalData, fakeStats, fakeData
        else:
            print('GOTCHA!!!')
            return estats, ecalDataPojo, torch.load(statsname, map_location='cpu'), \
                   domain.ecaldata.EcalData(torch.load(generatedName, map_location='cpu').reshape(shape).cpu().numpy(), ecalDataPojo.momentum, ecalDataPojo.point)

    ds4 = 'caloGAN_v4_case4_10K'
    estats4, edata4, gstats4, generated4 =  getStats(ds4)
    estats3, edata3, gstats3, generated3 =  getStats('caloGAN_v4_case3_10K')
    estats2, edata2, gstats2, generated2 =  getStats('caloGAN_v4_case2_50K')


    plotUi = PUI.ShowPlotUi(figsize=(8.9, 8.9)) #PUI.PDFPlotUi(outputfile)

    layout = 0

    def runSubplot(func, statType, pos, orth, range=True):
        pos = pos + 1
        plt.subplot(4,4, pos)
        func([
            estats4.get(statType, orth), gstats4.get(statType, orth),
            estats3.get(statType, orth), gstats3.get(statType, orth),
            estats2.get(statType, orth), gstats2.get(statType, orth)
        ], orth, range)
        return pos

    def runPlotLayoutSublots(position):
        position = runSubplot(doPlotAssymetryArr, AccumEnum.ASSYMETRY, position, False)
        position = runSubplot(doPlotAssymetryArr, AccumEnum.ASSYMETRY, position, True)
        position = runSubplot(doPlotShowerWidthArr, AccumEnum.WIDTH, position, False)
        position = runSubplot(doPlotShowerWidthArr, AccumEnum.WIDTH, position, True)
        position = runSubplot(plotEnergiesArr, AccumEnum.ENERGY, position, False)
        position = runSubplot(plotEnergiesArr, AccumEnum.ENERGY, position, True)
    #plotUi.toView(lambda: runPlotLayoutSublots(layout))

    def runPlotCLUSTERSHAPE(position):
        for crop in estats2.get(AccumEnum.CLUSTER_SHAPE, True):
            plotShape = lambda cs,o,r: plotEnergiesArr(list(map(lambda shapes: shapes.get(crop),cs)),o,r, crop)
            position = runSubplot(plotShape, AccumEnum.CLUSTER_SHAPE, position, False)
            position = runSubplot(plotShape, AccumEnum.CLUSTER_SHAPE, position, True)

    #plotUi.toView(lambda: runPlotCLUSTERSHAPE(layout))


    #runPlotMeans(ecalData, fakeData, haveFake, plotUi)

    def plotResponses(ecalData, fakeData, initLayout):
        combined = [ecalData, fakeData]
        vmin, vmax = np.amin(np.ma.masked_invalid(combined)), np.amax(combined)
        plt.subplot(3,4,initLayout)
        plotResponseImg(ecalData, vmin, vmax)
        initLayout+=1
        plt.subplot(3,4,initLayout)
        plotResponseImg(fakeData, vmin, vmax)
        initLayout+=1

        combined = np.log10(combined)
        vmin, vmax = np.amin(np.ma.masked_invalid(combined)), np.amax(combined)
        plt.subplot(3, 4, initLayout)
        plotResponseImg(combined[0], vmin, vmax)
        initLayout+=1
        plt.subplot(3,4, initLayout)
        plotResponseImg(combined[1], vmin, vmax)
        initLayout+=1

    def plotResps():
        plotResponses(edata4.response[7], generated4.response[10], 1)
        plotResponses(edata3.response[1], generated3.response[0], 5)
        plotResponses(edata2.response[6], generated2.response[16], 9)

    #plotUi = PUI.ShowPlotUi(figsize=(8.9, 6))
    #plotUi.toView(plotResps)

#conf()