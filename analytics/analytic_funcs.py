import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
from scipy.interpolate import interp2d
##include math
from analytics.plotAnalytics import doPlotAssymetry, doPlotShowerWidth, doPlotSparsity, plotMeanWithTitle, \
    plotMeanAbsDiff, plotResponses
from domain.ecaldata import parseEcalData
from plots.plotUi import PDFPlotUi
from os.path import dirname



def newline(p1, p2):      # функция отрисовки прямой
    ax = plt.gca()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        xmin, xmax = ax.get_xbound()
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    line = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(line)
    return line


def lineFunc(orthog, point, p): #p is momentum (0=x, 1=y, 2=z)
    x0, y0 = rotate(p, point)

    return doLineFunc(orthog, p, x0, y0)


def doLineFunc(orthog, p, x0, y0):
    mul = 1 if orthog else -1  # (x-x0)y/x + y0  or   -x(x-x0)/y + y0
    scale = mul * (p[0] ** -mul) * (p[1] ** mul)
    scaledX0fromY0 = scale * x0 + y0
    return lambda x: x * scale - scaledX0fromY0


def rotate(p, point):
    zoff = 25
    x0 = point[0] + zoff * p[0] / p[2]
    y0 = point[1] + zoff * p[1] / p[2]
    return x0, y0


def get_assymetry(data, ps, points, orthog=False):   # асимметрия ливня вдоль и поперек направнения наклона
    assym_res = []
    x = np.linspace(-14.5, 14.5, 30)
    y = np.linspace(-14.5, 14.5, 30)
    xx, yy = np.meshgrid(x, y)

    for i in range(len(data)):
        img = data[i]
        momentum = ps[i]
        point = points[i, :2]

        line_func = lineFunc(orthog, point, momentum)

        assym = computeAssym(img, line_func, momentum, orthog, xx, yy)
        assym_res.append(assym)

    return assym_res


def computeAssym(img, line_func, momentum, orthog, xx, yy):
    return doComputeAssym(img, line_func, momentum, orthog, xx, yy, np.sum(img))

def doComputeAssym(img, line_func, momentum, orthog, xx, yy, sumImg):
    idx = np.where(yy - line_func(xx) > 0) \
        if not orthog and momentum[1] < 0 \
        else np.where(yy - line_func(xx) < 0)
    zz = np.ones((30, 30))
    zz[idx] = 0
    return (np.sum(img * zz) - np.sum(img * (1 - zz))) / sumImg


def plotAssymetry(ecalData, orto, fakeData = None):
    assymetry_real = get_assymetry(ecalData.response, ecalData.momentum, ecalData.point, orto)
    assymetry_fake = get_assymetry(fakeData.response, fakeData.momentum, fakeData.point, orto) \
        if fakeData is not None else None

    doPlotAssymetry(assymetry_real, orto, assymetry_fake)
    return assymetry_real


def get_shower_width(response, momentum, points, orthog=False):      # ширина ливня вдоль и поперек направления
    spreads = []
    x = np.linspace(-14.5, 14.5, 30)
    y = np.linspace(-14.5, 14.5, 30)
    x_ = np.linspace(-14.5, 14.5, 100)

    for i in range(min(10000, len(response))):
        img = response[i]
        p = momentum[i]
        point = points[i]

        line_func = lineFunc(orthog, point, p)

        bb = interp2d(x, y, img, kind='cubic')

        y_ = line_func(x_)

        pyFracPx = p[1] / p[0]
        rescale = np.sqrt(1 + pyFracPx * pyFracPx)
        rescaledX = rescale * x_

        spread = computeWidth(bb, rescaledX, x_, y_)

        spreads.append(spread)
    return spreads


def computeWidth(bb, rescaledX, x_, y_):
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(100):
        ww = bb(x_[i], y_[i])
        if ww < 0: continue
        sum0 += ww
        rescaledXi = rescaledX[i]
        sum1 += rescaledXi * ww
        sum2 += rescaledXi * rescaledXi * ww
    sum1 = sum1 / sum0
    sum2 = sum2 / sum0
    if sum2 > sum1 * sum1:
        sigma = np.sqrt(sum2 - sum1 * sum1)
        spread = sigma[0]
    else:
        spread = 0
    return spread


def plotShowerWidth(ecalData, orto, fakeData = None):
    shower_width_real_direct = get_shower_width(ecalData.response, ecalData.momentum, ecalData.point, orto)
    fakeWidth = get_shower_width(fakeData.response, fakeData.momentum, fakeData.point, orto) \
        if fakeData is not None else None
    doPlotShowerWidth(shower_width_real_direct, orto, fakeWidth)
    return shower_width_real_direct


def get_ms_ratio2(img, ps, alpha=0.1):
    ms = np.sum(img)
    return computeMsRatio2(alpha, img, ms)


def computeMsRatio2(alpha, img, sumImg):
    ms_ = sumImg * alpha
    num = np.sum((img >= ms_))
    return num / 900.


def computeSparsity(response, momentum, alpha):
    sparsity = []
    for i in range(min(3000, len(response))):
        v_r = []
        for a in alpha:
            v_r.append(get_ms_ratio2(response[i], momentum[i], pow(10, a)))
        sparsity.append(v_r)
    return np.array(sparsity)


def plotSparsity(ecalData, fakeData=None):
    matplotlib.rcParams.update({'font.size': 14})
    # alpha = np.linspace(0, 0.02, 100)
    alpha = np.linspace(-5, 0, 50)

    sparsity = computeSparsity(ecalData.response, ecalData.momentum, alpha)
    fs = computeSparsity(fakeData.response, fakeData.momentum, alpha) \
        if fakeData is not None else None
    doPlotSparsity(sparsity, alpha, fs)


def runAnalytics(filename, ecalData, fakeData=None):
    if ecalData is None : ecalData = parseEcalData(filename)
    print(ecalData.title)

    plotUi = PDFPlotUi(dirname(dirname(__file__)) + '/resources/computed/' + filename + ('_generated' if fakeData is not None else '' + '.pdf'))  # ShowPlotUi()


    plotUi.toView(lambda: plotMeanWithTitle(ecalData.response, ecalData.title))
    if fakeData is not None :
        print(fakeData.title)
        plotUi.toView(lambda: plotMeanWithTitle(fakeData.response, fakeData.title))

        if (ecalData.response.shape[0] == fakeData.response.shape[0]) :
            plotUi.toView(lambda: plotMeanAbsDiff(ecalData.response, fakeData.response))

    plotUi.toView(lambda: plotResponses(ecalData, fakeData=fakeData))
    plotUi.toView(lambda: plotResponses(ecalData, False, fakeData))

    plotUi.toView(lambda: plotAssymetry(ecalData, False, fakeData))
    plotUi.toView(lambda: plotAssymetry(ecalData, True, fakeData))

    plotUi.toView(lambda: plotShowerWidth(ecalData, False, fakeData))
    plotUi.toView(lambda: plotShowerWidth(ecalData, True, fakeData))

    plotUi.toView(lambda: plotSparsity(ecalData, fakeData))

    plotUi.close()


def run():
    #runAnalytics('caloGAN_v3_case5_2K')
    dataset = 'caloGAN_v3_case4_2K'
    runAnalytics(dataset, ecalData = parseEcalData(dataset), fakeData=parseEcalData('caloGAN_v3_case5_2K'))