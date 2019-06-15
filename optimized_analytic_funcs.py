import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.backends.backend_pdf as pltpdf
import matplotlib
import os
import analytic_funcs as AF
from scipy.interpolate import interp2d
##include math



def runAnalytics(filename, ecalData, fakeData=None):
    if ecalData is None : ecalData = AF.parseEcalData(filename)
    print(ecalData.title)

    plotUi = AF.PDFPlotUi('computed/' + filename + ('_generated' if fakeData is not None else '' + '.pdf'))  # ShowPlotUi()


    plotUi.toView(lambda: AF.plotMeanWithTitle(ecalData.response, ecalData.title))

    fakeStats = None
    if fakeData is not None :
        print(fakeData.title)
        plotUi.toView(lambda: AF.plotMeanWithTitle(fakeData.response, fakeData.title))

        if (ecalData.response.shape[0] == fakeData.response.shape[0]) :
            plotUi.toView(lambda: AF.plotMeanAbsDiff(ecalData.response, fakeData.response))

        fakeStats = optimized_analityc(fakeData)

    ecalStats = optimized_analityc(ecalData)

    plotUi.toView(lambda: AF.plotResponses(ecalData, fakeData=fakeData))
    plotUi.toView(lambda: AF.plotResponses(ecalData, False, fakeData))

    #fix for fakeData is None
    plotUi.toView(lambda: AF.doPlotAssymetry(ecalStats.get(AccumEnum.ASSYMETRY).assymNonOrtho, False, fakeStats.get(AccumEnum.ASSYMETRY).assymNonOrtho))
    plotUi.toView(lambda: AF.doPlotAssymetry(ecalStats.get(AccumEnum.ASSYMETRY).assymOrtho, True, fakeStats.get(AccumEnum.ASSYMETRY).assymOrtho))

    plotUi.toView(lambda: AF.doPlotShowerWidth(ecalStats.get(AccumEnum.WIDTH).widthNonOrtho, False, fakeStats.get(AccumEnum.WIDTH).widthNonOrtho))
    plotUi.toView(lambda: AF.doPlotShowerWidth(ecalStats.get(AccumEnum.WIDTH).widthOrtho, True, fakeStats.get(AccumEnum.WIDTH).widthOrtho))

    plotUi.toView(lambda: AF.doPlotSparsity(ecalStats.get(AccumEnum.SPARSITY).sparsity, fakeStats.get(AccumEnum.SPARSITY).sparsity))

    plotUi.close()


def run():
    #runAnalytics('caloGAN_v3_case5_2K')
    dataset = 'caloGAN_v3_case4_2K'
    runAnalytics(dataset, ecalData = AF.parseEcalData(dataset), fakeData=AF.parseEcalData('caloGAN_v3_case5_2K'))


class AssymetryAccumulator :
    def __init__(self, imgSize) -> None:
        self.imgSize = imgSize
        self.assymOrtho = []
        self.assymNonOrtho = []
        x = np.linspace(-14.5, 14.5, imgSize)
        y = np.linspace(-14.5, 14.5, imgSize)
        self.xx, self.yy = np.meshgrid(x, y)

    def append(self, momentum, img, lineOrt, lineNotOrt, sumImg):
        self.assymOrtho.append(AF.doComputeAssym(img, lineOrt, momentum, True, self.xx, self.yy, sumImg))
        self.assymNonOrtho.append(AF.doComputeAssym(img, lineNotOrt, momentum, True, self.xx, self.yy, sumImg))


class ShowerWidthAccumulator :
    def __init__(self, imgSize) -> None:
        self.imgSize = imgSize
        self.widthOrtho = []
        self.widthNonOrtho = []
        self.x = np.linspace(-14.5, 14.5, imgSize)
        self.y = np.linspace(-14.5, 14.5, imgSize)
        self.x_ = np.linspace(-14.5, 14.5, 100)
        self.interpolation = lambda img: interp2d(self.x, self.y, img, kind='cubic')

    def append(self, momentum, img, lineOrt, lineNotOrt):
        bb = self.interpolation(img)

        y_Ort = lineOrt(self.x_)
        y_NotOrt = lineNotOrt(self.x_)

        pyFracPx = momentum[1] / momentum[0]
        rescale = np.sqrt(1 + pyFracPx * pyFracPx)
        rescaledX = rescale * self.x_

        self.widthOrtho.append(AF.computeWidth(bb, rescaledX, self.x_, y_Ort))
        self.widthNonOrtho.append(AF.computeWidth(bb, rescaledX, self.x_, y_NotOrt))


class SparsityAccumulator :
    def __init__(self) -> None:
        self.alpha = np.linspace(-5, 0, 50)
        self.sparsity = []

    def append(self, momentum, img, sumImg):
        v_r = []
        for a in self.alpha:
            v_r.append(AF.computeMsRatio2(pow(10, a), img, sumImg))
        self.sparsity.append(v_r)

class AccumEnum :
    ASSYMETRY = "A"
    WIDTH = "W"
    SPARSITY = "S"


def optimized_analityc(ecalData) :
    response = ecalData.response
    momentum = ecalData.momentum
    points = ecalData.point

    assym = AssymetryAccumulator(30)
    width = ShowerWidthAccumulator(30)
    sprsity = SparsityAccumulator()

    for i in range(len(response)) :
        img = response[i]
        p = momentum[i]
        point = points[i]

        sumImg = np.sum(img)

        x0, y0 = AF.rotate(p, point)

        lfOrthog = AF.doLineFunc(True, p, x0, y0)
        lfNotOrthog = AF.doLineFunc(False, p, x0, y0)

        assym.append(p, img, lfOrthog, lfNotOrthog, sumImg)
        if i <= 10000 :
            width.append(p, img, lfOrthog, lfNotOrthog)
        if i <= 3000 :
            sprsity.append(p, img, sumImg)

    return {AccumEnum.ASSYMETRY : assym,
            AccumEnum.WIDTH : width,
            AccumEnum.SPARSITY : sprsity}