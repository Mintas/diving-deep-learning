import numpy as np
from os.path import dirname

import analytics.plotAnalytics
from analytics.plotAnalytics import doPlotAssymetry, doPlotShowerWidth, doPlotSparsity, plotMeanWithTitle, \
    plotMeanAbsDiff, plotResponses, plotEnergies
from analytics import analytic_funcs as AF
from scipy.interpolate import interp2d
##include math
import domain.ecaldata as ED
import plots.plotUi as PUI

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


class ShowerWidthAccumulator :
    def __init__(self, imgSize) -> None:
        self.imgSize = imgSize
        self.widthOrtho = []
        self.widthNonOrtho = []
        bound = imgSize/2.0 - 0.5
        self.x = np.linspace(-bound, bound, imgSize)
        self.y = np.linspace(-bound, bound, imgSize)
        self.x_ = np.linspace(-bound, bound, 100)
        #self.interpolation = lambda img: interp2d(self.x, self.y, img, kind='cubic')

    def append(self, momentum, img, lineOrt, lineNotOrt):
        bb = self.interpolation(img)

        y_Ort = lineOrt(self.x_)
        y_NotOrt = lineNotOrt(self.x_)

        pyFracPx = momentum[1] / momentum[0]
        rescale = np.sqrt(1 + pyFracPx * pyFracPx)
        rescaledX = rescale * self.x_

        self.widthOrtho.append(AF.computeWidth(bb, rescaledX, self.x_, y_Ort))
        self.widthNonOrtho.append(AF.computeWidth(bb, rescaledX, self.x_, y_NotOrt))

    def interpolation(self, img):
        return interp2d(self.x, self.y, img, kind='cubic')


# class ShiftAccumulator :
#     def __init__(self, imgSize) -> None:
#         self.imgSize = imgSize
#         self.shifts = []
#         self.zoff = 25.
#         bound = imgSize/2.0 - 0.5
#         x = np.linspace(-bound, bound, imgSize)
#         y = np.linspace(-bound, bound, imgSize)
#         self.xx, self.yy = np.meshgrid(x, y)
#
#     def append(self, momentum, img, lineOrt, lineNotOrt, sumImg):
#         self.assymOrtho.append(AF.doComputeAssym(img, lineOrt, momentum, True, self.xx, self.yy, sumImg))
#         self.assymNonOrtho.append(AF.doComputeAssym(img, lineNotOrt, momentum, True, self.xx, self.yy, sumImg))


class SparsityAccumulator :
    def __init__(self) -> None:
        self.alpha = np.linspace(-5, 0, 50)
        self.sparsity = []

    def append(self, momentum, img, sumImg):
        v_r = []
        for a in self.alpha:
            v_r.append(AF.computeMsRatio2(pow(10, a), img, sumImg))
        self.sparsity.append(v_r)

class EnergyResponseAccumulator :
    def __init__(self) -> None:
        self.energies = []

    def append(self, momentum, sumImg):
        self.energies.append(sumImg)

class AccumEnum :
    ASSYMETRY = "A"
    WIDTH = "W"
    SPARSITY = "S"
    ENERGY = "ER"


def optimized_analityc(ecalData, imgsize) :
    response = ecalData.response
    momentum = ecalData.momentum
    points = ecalData.point

    assym = AssymetryAccumulator(imgsize)
    width = ShowerWidthAccumulator(imgsize)
    sprsity = SparsityAccumulator()
    nrgy = EnergyResponseAccumulator()

    for i in range(len(response)) :
        img = response[i]
        p = momentum[i]
        point = points[i]

        sumImg = np.sum(img)

        x0, y0 = AF.rotate(p, point)

        lfOrthog = AF.doLineFunc(True, p, x0, y0)
        lfNotOrthog = AF.doLineFunc(False, p, x0, y0)

        nrgy.append(p, sumImg)
        assym.append(p, img, lfOrthog, lfNotOrthog, sumImg)
        if i <= 10000 :
            width.append(p, img, lfOrthog, lfNotOrthog)
        if i <= 3000 :
            sprsity.append(p, img, sumImg)

    return {AccumEnum.ASSYMETRY : assym,
            AccumEnum.WIDTH : width,
            AccumEnum.SPARSITY : sprsity,
            AccumEnum.ENERGY : nrgy}

def runAnalytics(filename, ecalData, fakeData=None, ecalStats=None, fakeStats=None):
    print(ecalData.title)

    haveFake = fakeData is not None
    plotUi = PUI.PDFPlotUi(dirname(dirname(__file__)) + filename + '_stats' + ('_generated' if haveFake else ''))  # ShowPlotUi()

    plotUi.toView(lambda: plotMeanWithTitle(ecalData.response, ecalData.title))

    plotUi.toView(lambda: plotResponses(ecalData, fakeData=fakeData))
    plotUi.toView(lambda: plotResponses(ecalData, False, fakeData))

    imgSize = ecalData.response[0].shape[0]
    if haveFake:
        print(fakeData.title)
        plotUi.toView(lambda: plotMeanWithTitle(fakeData.response, fakeData.title))

        if (ecalData.response.shape[0] == fakeData.response.shape[0]) :
            plotUi.toView(lambda: plotMeanAbsDiff(ecalData.response, fakeData.response))

        if fakeStats is None :
            fakeStats = optimized_analityc(fakeData, imgSize)
    if ecalStats is None :
        ecalStats = optimized_analityc(ecalData, imgSize)

    plotUi.toView(lambda: doPlotAssymetry(ecalStats.get(AccumEnum.ASSYMETRY).assymNonOrtho, False, fakeStats.get(AccumEnum.ASSYMETRY).assymNonOrtho) if haveFake else None)
    plotUi.toView(lambda: doPlotAssymetry(ecalStats.get(AccumEnum.ASSYMETRY).assymOrtho, True, fakeStats.get(AccumEnum.ASSYMETRY).assymOrtho) if haveFake else None)

    plotUi.toView(lambda: doPlotShowerWidth(ecalStats.get(AccumEnum.WIDTH).widthNonOrtho, False, fakeStats.get(AccumEnum.WIDTH).widthNonOrtho) if haveFake else None)
    plotUi.toView(lambda: doPlotShowerWidth(ecalStats.get(AccumEnum.WIDTH).widthOrtho, True, fakeStats.get(AccumEnum.WIDTH).widthOrtho) if haveFake else None)

    es = ecalStats.get(AccumEnum.SPARSITY).sparsity
    plotUi.toView(lambda: doPlotSparsity(es, ecalStats.get(AccumEnum.SPARSITY).alpha, fakeStats.get(AccumEnum.SPARSITY).sparsity) if haveFake else None)

    plotUi.toView(lambda: plotEnergies(ecalStats.get(AccumEnum.ENERGY).energies, fakeStats.get(AccumEnum.ENERGY).energies) if haveFake else None)

    plotUi.close()


def run():
    #runAnalytics('caloGAN_v3_case5_2K')
    dataset = 'caloGAN_v3_case4_2K'
    runAnalytics(dataset, ecalData = ED.parseEcalData(dataset))#, fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
#run()