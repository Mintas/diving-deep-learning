import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.backends.backend_pdf as pltpdf
import matplotlib
import os
from scipy.interpolate import interp2d
##include math


def plotMean(imgs):
    plt.imshow(np.mean(imgs, axis=0, keepdims=False))  # средняя картинка по реальному датасету
    plt.colorbar()

def plotMeanWithTitle(imgs, title):
    plt.title(title)
    plotMean(imgs)

def plotMeanAbsDiff(ecal, fake) :
    plt.imshow(abs(np.mean(ecal, axis=0, keepdims=False) - np.mean(fake, axis=0, keepdims=False)))
    plt.colorbar()

def plotLogResponse(img, logScale):
    plt.imshow(np.log10(img)) if logScale else plt.imshow(img)
    # #todo : what to do if we got 0 as one of inputs here?
    plt.colorbar()
    plt.xlabel('cell X \n')
    plt.ylabel('cell Y')


def plotResponses(ecalData, logScale=True, fakeData = None):
    matplotlib.rcParams.update({'font.size': 14})
    for i in range(4):
        #print("im:", i)
        #print("momentum", real_p[i])
        # print ("fake", fake_p[i])
        plt.subplot(521 + 2*i)  # todo : how to update this correctly ?
        plotLogResponse(ecalData.response[i], logScale)
        plt.subplot(521 + 2*i+1)  # todo : how to update this correctly ?
        asFake = fakeData.response[i] if fakeData is not None else ecalData.response[-i]
        plotLogResponse(asFake, logScale)


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

def doPlotAssymetry(assymetry_real, orto, assymetry_fake = None):
    plt.hist(assymetry_real, bins=50, range=[-1, 1], color='red', alpha=0.3, density=True, label='Geant')
    if assymetry_fake is not None:
        plt.hist(assymetry_fake,bins=50, range=[-1, 1], color='blue', alpha=0.3, density=True, label='GAN')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster asymmetry')
    plt.legend(loc='best')


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

def doPlotShowerWidth(ecalWidth, orto, fakeWidth = None):
    matplotlib.rcParams.update({'font.size': 14})
    plt.hist(ecalWidth, bins=50, range=[0, 15], density=True, alpha=0.3, color='red', label='Geant')

    if fakeWidth is not None:
        plt.hist(fakeWidth, bins=50, range=[0, 15], density=True, alpha=0.3, color='blue', label='GAN');
    plt.legend(loc='best')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster width [cm]')
    plt.ylabel('Arbitrary units')

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

def doPlotSingleSparsity(sparsity, alpha, color='red'):
    means = np.mean(sparsity, axis=0)
    stddev = np.std(sparsity, axis=0)
    plt.plot(alpha, means, color=color)
    plt.fill_between(alpha, means - stddev, means + stddev, color=color, alpha=0.3)
    #plt.plot(alpha, means_f, color='blue')
    #plt.fill_between(alpha, means_f - stddev_f, means_f + stddev_f, color='blue', alpha=0.3)



def plotSparsity(ecalData, fakeData=None):
    matplotlib.rcParams.update({'font.size': 14})
    # alpha = np.linspace(0, 0.02, 100)
    alpha = np.linspace(-5, 0, 50)

    sparsity = computeSparsity(ecalData.response, ecalData.momentum, alpha)
    fs = computeSparsity(fakeData.response, fakeData.momentum, alpha) \
        if fakeData is not None else None
    doPlotSparsity(sparsity, fs)

def doPlotSparsity(ecalSparsity, alpha, fakeSparsity=None):
    matplotlib.rcParams.update({'font.size': 14})

    doPlotSingleSparsity(ecalSparsity, alpha)
    legend = ['Geant']
    if fakeSparsity is not None:
        doPlotSingleSparsity(fakeSparsity, alpha, color='blue')
        legend.append('GAN')

    plt.legend(legend)
    plt.title('Sparsity')
    plt.xlabel('log10(Threshold/GeV)')
    plt.ylabel('Fraction of cells above threshold')




class ShowPlotUi():
    def toView(self, plot):
        plot()
        plt.show()

    def close(self):
        pass


class PDFPlotUi():
    def __init__(self, pdfFile) -> None:
        i = 0
        lookupNextName = pdfFile
        while (os.path.isfile(lookupNextName + '.pdf')) :
            i = i + 1
            lookupNextName = pdfFile + '_' + str(i)
        self.pdf = matplotlib.backends.backend_pdf.PdfPages(lookupNextName + '.pdf')

    def toView(self, plot):
        fig = plt.figure(figsize=(10, 10))
        plot()
        self.pdf.savefig(fig)

    def close(self):
        self.pdf.close()

class EcalData :
    def __init__(self, response, momentum, point, title='') -> None:
        self.response = response
        self.momentum = momentum
        self.point = point
        self.title = title

def parseEcalData(filename):
    ecal = np.load('ecaldata/' + filename + '.npz')
    ecal.keys()
    return dictToEcalData(ecal)


def dictToEcalData(ecal):
    real_imgs = ecal['EnergyDeposit']
    real_p = ecal['ParticleMomentum']
    real_point = ecal['ParticlePoint']
    title = 'EnergyDeposit  shape: ' + str(real_imgs.shape) + \
            '\n min: ' + str(real_imgs.min()) + '\t max: ' + str(real_imgs.max()) + \
            '\n first particle Momentum :  ' + str(real_p[0]) + \
            '\n Point :' + str(real_point[0]) + \
            '\n particle type is : ' + str(ecal['ParticlePDG'][0])
    return EcalData(real_imgs, real_p, real_point, title)


def runAnalytics(filename, ecalData, fakeData=None):
    if ecalData is None : ecalData = parseEcalData(filename)
    print(ecalData.title)

    plotUi = PDFPlotUi('computed/' + filename + ('_generated' if fakeData is not None else '' + '.pdf'))  # ShowPlotUi()


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