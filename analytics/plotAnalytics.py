import matplotlib
import numpy as np
from matplotlib import pyplot as plt

def plotMean(imgs):
    plt.imshow(np.mean(imgs, axis=0, keepdims=False))  # средняя картинка по реальному датасету
    plt.colorbar()


def plotMeanWithTitle(imgs, title):
    plt.title(title)
    plotMean(imgs)


def plotMeanAbsDiff(ecal, fake) :
    plt.imshow(abs(np.mean(ecal, axis=0, keepdims=False) - np.mean(fake, axis=0, keepdims=False)))
    plt.colorbar()


def plotLogResponse(img, vmin=None, vmax=None):
    plt.imshow(img, interpolation='nearest', vmin=vmin, vmax=vmax)
    # #todo : what to do if we got 0 as one of inputs here?
    plt.colorbar()
    #plt.xlabel('cell X \n')
    plt.ylabel('cell Y')
    plt.xticks([])
    plt.yticks([])

def plotResponses(ecalData, logScale=True, fakeData = None):
    matplotlib.rcParams.update({'font.size': 14})

    combined = np.append(ecalData.response[:4], fakeData.response[:4]) if fakeData is not None else ecalData.response[:8]
    if logScale: combined = np.log10(combined)
    vmin, vmax = np.amin(combined), np.amax(combined)

    for (i, response) in enumerate(combined):
        plt.subplot(521 + i)  # todo : how to update this correctly ?
        plotLogResponse(response, vmin, vmax)

def doPlotAssymetry(assymetry_real, orto, assymetry_fake = None, range=None):
    postfix = ', narrow hist'
    if range is None: range, postfix = [-1, 1], ''
    plt.hist(assymetry_real, bins=50, range=range, color='red', alpha=0.3, density=True, label='Geant')
    if assymetry_fake is not None:
        plt.hist(assymetry_fake,bins=50, range=range, color='blue', alpha=0.3, density=True, label='GAN')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster asymmetry' + postfix)
    plt.legend(loc='best')

def plotAssymWithNpHist(ecalAssym): #currently unused, represents template for stats calc
    #1st step : calculate histogram
    assymHist, binz = np.histogram(ecalAssym, bins=50, range=[-1, 1], density=True)
    #2nd step : plot hist by weights; is equal to plt.hist(ecalAssym, bins=.....)
    plt.hist(binz[:-1], bins=len(binz), weights=assymHist, range=[-1, 1], color='red', alpha=0.3, density=True, label='Geant')

def doPlotShowerWidth(ecalWidth, orto, fakeWidth = None, range=None):
    postfix = ', narrow hist'
    if range is None: range, postfix = [0,15], ''
    matplotlib.rcParams.update({'font.size': 14})
    plt.hist(ecalWidth, bins=50, range=range, density=True, alpha=0.3, color='red', label='Geant')

    if fakeWidth is not None:
        plt.hist(fakeWidth, bins=50, range=range, density=True, alpha=0.3, color='blue', label='GAN')
    plt.legend(loc='best')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster width [cm]' + postfix)
    plt.ylabel('Arbitrary units')


def doPlotSingleSparsity(sparsity, alpha, color='red'):
    means = np.mean(sparsity, axis=0)
    stddev = np.std(sparsity, axis=0)
    plt.plot(alpha, means, color=color)
    plt.fill_between(alpha, means - stddev, means + stddev, color=color, alpha=0.3)


def doPlotSparsity(ecalSparsity, alpha, fakeSparsity=None):
    matplotlib.rcParams.update({'font.size': 14})

    doPlotSingleSparsity(np.array(ecalSparsity), alpha)
    legend = ['Geant']
    if fakeSparsity is not None:
        doPlotSingleSparsity(np.array(fakeSparsity), alpha, color='blue')
        legend.append('GAN')

    plt.legend(legend)
    plt.title('Sparsity')
    plt.xlabel('log10(Threshold/GeV)')
    plt.ylabel('Fraction of cells above threshold')


def plotEnergies(ecal, logScaled, fake=None, rangeByExpectedOnly=True):
    maxE = max(ecal)
    minE = min(ecal)
    if fake is not None and not rangeByExpectedOnly:
        maxE = max(maxE, max(fake))
        minE = min(minE, min(fake))

    plt.hist(ecal, 100, range=[minE, maxE], log=logScaled, color='red', alpha=0.3)
    legend = ['Geant']

    if fake is not None:
        plt.hist(fake, 100, range=[minE, maxE], log=logScaled, color='blue', alpha=0.3)
        legend.append('GAN')

    plt.legend(legend)
    plt.xlabel('Energy')
    plt.ylabel('Arbitrary Units' + ', LogScale' if logScaled else '')