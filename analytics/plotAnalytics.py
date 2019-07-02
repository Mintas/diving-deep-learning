import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import analytics.analytic_funcs as AF

def plotMean(imgs):
    plt.imshow(np.mean(imgs, axis=0, keepdims=False))  # средняя картинка по реальному датасету
    plt.colorbar()


def plotMeanWithTitle(imgs, title):
    matplotlib.rcParams.update({'font.size': 13})
    plt.suptitle('ECAL: mean response of deposited energy', fontsize=16)
    plt.title(title)
    plotMean(imgs)


def plotMeanAbsDiff(ecal, fake) :
    matplotlib.rcParams.update({'font.size': 13})
    plt.imshow(abs(np.mean(ecal, axis=0, keepdims=False) - np.mean(fake, axis=0, keepdims=False)))
    plt.colorbar()


def plotResponseImg(img, vmin=None, vmax=None, removeTicks=True):
    matplotlib.rcParams.update({'font.size': 13})
    plt.imshow(img, interpolation='nearest', vmin=vmin, vmax=vmax)
    # #todo : what to do if we got 0 as one of inputs here?
    plt.colorbar()
    #plt.xlabel('cell X \n')
    if removeTicks:
        plt.ylabel('cell Y')
        plt.xticks([])
        plt.yticks([])

def plotResponses(ecalData, logScale=True, fakeData = None):
    depositeSupTitle = 'Examples of ' + ('logScaled ' if logScale else '') + 'energy deposite \n generated by GEANT4' \
                       + ('' if fakeData is None else '(left) vs. generated by GAN(right)')
    plt.suptitle(depositeSupTitle, fontsize=16)
    matplotlib.rcParams.update({'font.size': 13})

    combined = np.concatenate((ecalData.response[:4], fakeData.response[:4])) if fakeData is not None else ecalData.response[:8]
    if logScale: combined = np.log10(combined)
    vmin, vmax = np.amin(np.ma.masked_invalid(combined)), np.amax(combined)

    for i in range(8):
        plt.subplot(421 + i)  # todo : how to update this correctly ?
        plotResponseImg(combined[4 * (i % 2) + i // 2], vmin, vmax)
        #plotResponseImg(combined[4 * (i % 2) + i // 2], None, None)

def doPlotAssymetry(assymetry_real, orto, assymetry_fake = None, rangeByReal=True):
    range, postfix = comonHistRange(assymetry_real, assymetry_fake, rangeByReal)
    histReal, binz, patches = plt.hist(assymetry_real, bins=50, range=range, color='red', alpha=0.3, density=True, label='Geant')
    if assymetry_fake is not None:
        histFake,_,_ = plt.hist(assymetry_fake,bins=50, range=range, color='blue', alpha=0.3, density=True, label='GAN')
        postfix += '\n' + statsMsg(histFake, histReal)
    plt.xlabel(('Longitudinal' if orto else 'Transverse') + ' cluster asymmetry' + postfix)
    plt.legend(loc='best')


def statsMsg(histFake, histReal):
    #chiStat, pVal = AF.chiSquare(histReal, histFake)  #todo : how to compute chiSquare stats for matplotlib/numpy hist ?
    #chiMsg = 'ChiSqr: { stats = %d , p-val = %d }' % (chiStat, pVal)
    l1msg  = 'L1 dist.: {:10.4f} '.format(AF.l1norm(histReal, histFake))
    l2msg  = 'L2 dist.: {:10.4f} '.format(AF.l2norm(histReal, histFake))
    return l1msg + ' ; ' + l2msg

def plotAssymWithNpHist(ecalAssym): #currently unused, represents template for stats calc
    #1st step : calculate histogram
    assymHist, binz = np.histogram(ecalAssym, bins=50, range=[-1, 1], density=True)
    #2nd step : plot hist by weights; is equal to plt.hist(ecalAssym, bins=.....)
    plt.hist(binz[:-1], bins=len(binz), weights=assymHist, range=[-1, 1], color='red', alpha=0.3, density=True, label='Geant')

def doPlotShowerWidth(ecalWidth, orto, fakeWidth = None, rangeByReal=True):
    range, postfix = comonHistRange(ecalWidth, fakeWidth, rangeByReal)
    matplotlib.rcParams.update({'font.size': 13})

    histReal,_,_ = plt.hist(ecalWidth, bins=50, range=range, density=True, alpha=0.3, color='red', label='Geant')

    if fakeWidth is not None:
        histFake,_,_ = plt.hist(fakeWidth, bins=50, range=range, density=True, alpha=0.3, color='blue', label='GAN')
        postfix += '\n' + statsMsg(histFake, histReal)
    plt.legend(loc='best')
    plt.xlabel(('Longitudinal' if orto else 'Transverse') + ' cluster width [cm]' + postfix)
    plt.ylabel('Arbitrary units')


def doPlotSingleSparsity(sparsity, alpha, color='red'):
    means = np.mean(sparsity, axis=0)
    stddev = np.std(sparsity, axis=0)
    plt.plot(alpha, means, color=color)
    plt.fill_between(alpha, means - stddev, means + stddev, color=color, alpha=0.3)


def doPlotSparsity(ecalSparsity, alpha, fakeSparsity=None):
    matplotlib.rcParams.update({'font.size': 13})

    doPlotSingleSparsity(np.array(ecalSparsity), alpha)
    legend = ['Geant']
    if fakeSparsity is not None:
        doPlotSingleSparsity(np.array(fakeSparsity), alpha, color='blue')
        legend.append('GAN')

    plt.legend(legend)
    plt.title('Sparsity')
    plt.xlabel('log10(Threshold/GeV)')
    plt.ylabel('Fraction of cells above threshold')


def plotEnergies(ecal, logScaled, fake=None, rangeByReal=True, size=30):
    plt.suptitle('Energy deposited in {}x{} '.format(size, size), fontsize=16)
    yPostfix = ', LogScale' if logScaled else ''
    commonRange, xPostfix = comonHistRange(ecal, fake, rangeByReal)

    histReal,_,_ = plt.hist(ecal, 100, range=commonRange, log=logScaled, color='red', alpha=0.3)
    legend = ['Geant']

    if fake is not None:
        histFake,_,_ = plt.hist(fake, 100, range=commonRange, log=logScaled, color='blue', alpha=0.3)
        if logScaled :
            histFake[histFake == 0] = 1
            histReal[histReal == 0] = 1
            histFake, histReal = np.log10(histFake), np.log10(histReal)
        xPostfix += '\n' + statsMsg(histFake, histReal)
        legend.append('GAN')

    plt.legend(legend)
    plt.xlabel('Energy ' + xPostfix)
    plt.ylabel('Arbitrary Units' + yPostfix)


def comonHistRange(ecal, fake, rangeByReal):
    maxE = np.amax(ecal)
    minE = np.amin(ecal)
    if fake is not None and not rangeByReal:
        maxE = max(maxE, np.amax(fake))
        minE = min(minE, np.amin(fake))
    return [minE, maxE], '' if rangeByReal else ', common range'