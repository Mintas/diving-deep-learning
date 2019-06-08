import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.backends.backend_pdf as pltpdf
import matplotlib
import os
from scipy.interpolate import interp2d
##include math


def plotMean(imgs):
    plt.imshow(np.mean(imgs, axis=0))  # средняя картинка по реальному датасету
    plt.colorbar()

def plotMeanWithTitle(imgs, title):
    plt.title(title)
    plotMean(imgs)

def plotLogResponse(img, logScale):
    plt.imshow(np.log10(img)) if logScale else plt.imshow(img)
    # #todo : what to do if we got 0 as one of inputs here?
    plt.colorbar()
    plt.xlabel('cell X \n')
    plt.ylabel('cell Y')


def plotResponses(real_imgs, count, real_p, logScale=True):
    matplotlib.rcParams.update({'font.size': 14})
    for i in range(count):
        #print("im:", i)
        #print("momentum", real_p[i])
        # print ("fake", fake_p[i])
        plt.subplot(521 + i)  # todo : how to update this correctly ?
        plotLogResponse(real_imgs[i], logScale)


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
    zoff = 25
    point0 = point[0] + zoff * p[0] / p[2]
    point1 = point[1] + zoff * p[1] / p[2]

    mul = 1 if orthog else -1  # (x-x0)y/x + y0  or   -x(x-x0)/y + y0
    line_func = lambda x: mul * (x - point0) * (p[0] ** -mul) * (p[1] ** mul) + point1
    return line_func


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

        idx = np.where(yy - line_func(xx) < 0)
        if (not orthog and momentum[1] < 0):
            idx = np.where(yy - line_func(xx) > 0)

        zz = np.ones((30, 30))
        zz[idx] = 0

        assym = (np.sum(img * zz) - np.sum(img * (1 - zz))) / np.sum(img)
        assym_res.append(assym)

    return assym_res


def plotAssymetry(real_imgs, real_p, real_point, orto):
    assymetry_real = get_assymetry(real_imgs, real_p, real_point, orto)
    plt.hist(assymetry_real, bins=50, range=[-1, 1], color='red', alpha=0.3, density=True, label='Geant')
    # plt.hist(assymetry_fake, bins=50, range=[-1, 1], color='blue', alpha=0.3, density=True, label='GAN');
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster asymmetry')
    plt.legend(loc='best')
    #plt.show()
    return assymetry_real


def get_shower_width(data, ps, points, orthog=False):      # ширина ливня вдоль и поперек направления
    spreads = []
    x = np.linspace(-14.5, 14.5, 30)
    y = np.linspace(-14.5, 14.5, 30)
    x_ = np.linspace(-14.5, 14.5, 100)

    for i in range(min(10000, len(data))):
        img = data[i]
        p = ps[i]
        point = points[i]

        line_func = lineFunc(orthog, point, p)

        bb = interp2d(x, y, img, kind='cubic')

        y_ = line_func(x_)

        pyFracPx = p[1] / p[0]
        rescale = np.sqrt(1 + pyFracPx * pyFracPx)

        sum0 = 0
        sum1 = 0
        sum2 = 0
        for i in range(100):
            ww = bb(x_[i], y_[i])
            if ww < 0: continue
            sum0 += ww
            rescaledX = rescale * x_[i]
            sum1 += rescaledX * ww
            sum2 += rescaledX * rescaledX * ww

        sum1 = sum1 / sum0
        sum2 = sum2 / sum0
        if sum2 >= sum1 * sum1:
            sigma = np.sqrt(sum2 - sum1 * sum1)
            spreads.append(sigma[0])
        else:
            spreads.append(0)

    return spreads


def plotShowerWidth(real_imgs, real_p, real_point, orto):
    shower_width_real_direct = get_shower_width(real_imgs, real_p, real_point, orto)
    matplotlib.rcParams.update({'font.size': 14})
    plt.hist(shower_width_real_direct, bins=50, range=[0, 15], density=True, alpha=0.3, color='red', label='Geant')
    # plt.hist(shower_width_fake_direct, bins=50, range=[0, 15], density=True, alpha=0.3, color='blue', label='GAN');
    plt.legend(loc='best')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster width [cm]')
    plt.ylabel('Arbitrary units')
    #plt.show()
    return shower_width_real_direct


def get_ms_ratio2(img, ps, alpha=0.1):
    ms = np.sum(img)
    ms_ = ms * alpha

    num = np.sum((img >= ms_))
    return num / 900.


def computeSparsity(real_imgs, real_p, alpha):
    sparsity = []
    for i in range(min(3000, len(real_imgs))):
        v_r = []
        for a in alpha:
            v_r.append(get_ms_ratio2(real_imgs[i], real_p[i], pow(10, a)))
        sparsity.append(v_r)
    return np.array(sparsity)

def doPlotSparsity(means, stddev, alpha):
    matplotlib.rcParams.update({'font.size': 14})
    plt.plot(alpha, means, color='red')
    plt.fill_between(alpha, means - stddev, means + stddev, color='red', alpha=0.3)
    #plt.plot(alpha, means_f, color='blue')
    #plt.fill_between(alpha, means_f - stddev_f, means_f + stddev_f, color='blue', alpha=0.3)
    plt.legend(['Geant']) #, 'GAN'])
    plt.title('Sparsity')
    plt.xlabel('log10(Threshold/GeV)')
    plt.ylabel('Fraction of cells above threshold')


def plotSparsity(real_imgs, real_p):
    # alpha = np.linspace(0, 0.02, 100)
    alpha = np.linspace(-5, 0, 50)

    sparsity = computeSparsity(real_imgs, real_p, alpha)
    means_r = np.mean(sparsity, axis=0)
    stddev_r = np.std(sparsity, axis=0)

    doPlotSparsity(means_r, stddev_r, alpha)


class ShowPlotUi():
    def toView(self, plot):
        plot()
        plt.show()

    def close(self):
        pass


class PDFPlotUi():
    def __init__(self, pdfFile) -> None:
        i = 0
        while (os.path.isfile(pdfFile + '.pdf')) :
            i = i + 1
            pdfFile = pdfFile + '_' + str(i)
        self.pdf = matplotlib.backends.backend_pdf.PdfPages(pdfFile + '.pdf')

    def toView(self, plot):
        fig = plt.figure(figsize=(10, 10))
        plot()
        self.pdf.savefig(fig)

    def close(self):
        self.pdf.close()


def runAnalytics(filename):
    ecal = np.load('ecaldata/' + filename + '.npz')
    ecal.keys()
    real_imgs = ecal['EnergyDeposit']
    real_p = ecal['ParticleMomentum']
    real_point = ecal['ParticlePoint']

    plotUi = PDFPlotUi('computed/' + filename + '.pdf')  # ShowPlotUi()

    title = 'EnergyDeposit  shape: ' + str(real_imgs.shape) + \
            '\n min: ' + str(real_imgs.min()) + '\t max: ' + str(real_imgs.max()) + \
            '\n first particle Momentum :  ' + str(real_p[0]) + \
            '\n Point :' + str(real_point[0]) + \
            '\n particle type is : ' + str(ecal['ParticlePDG'][0])
    print(title)

    plotUi.toView(lambda: plotMeanWithTitle(real_imgs, title))

    plotUi.toView(lambda: plotResponses(real_imgs, 9, real_p))
    plotUi.toView(lambda: plotResponses(real_imgs, 9, real_p, False))

    plotUi.toView(lambda: plotAssymetry(real_imgs, real_p, real_point, False))
    plotUi.toView(lambda: plotAssymetry(real_imgs, real_p, real_point, True))

    plotUi.toView(lambda: plotShowerWidth(real_imgs, real_p, real_point, False))
    plotUi.toView(lambda: plotShowerWidth(real_imgs, real_p, real_point, True))

    plotUi.toView(lambda: plotSparsity(real_imgs, real_p))

    plotUi.close()


def run():
    #runAnalytics('caloGAN_v3_case5_2K')
    runAnalytics('caloGAN_v3_case4_2K')
    runAnalytics('caloGAN_v3_case3_2K')
    runAnalytics('caloGAN_v3_case2_50K')
    runAnalytics('caloGAN_v3_case1_50K')

