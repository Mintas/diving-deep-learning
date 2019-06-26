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


def plotLogResponse(img, logScale):
    plt.imshow(np.log10(img) if logScale else img, interpolation='nearest')
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
        plt.subplot(521 + 2*i+1)
        asFake = fakeData.response[i] if fakeData is not None else ecalData.response[-i]
        plotLogResponse(asFake, logScale)


def doPlotAssymetry(assymetry_real, orto, assymetry_fake = None):
    plt.hist(assymetry_real, bins=50, range=[-1, 1], color='red', alpha=0.3, density=True, label='Geant')
    if assymetry_fake is not None:
        plt.hist(assymetry_fake,bins=50, range=[-1, 1], color='blue', alpha=0.3, density=True, label='GAN')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster asymmetry')
    plt.legend(loc='best')


def doPlotShowerWidth(ecalWidth, orto, fakeWidth = None):
    matplotlib.rcParams.update({'font.size': 14})
    plt.hist(ecalWidth, bins=50, range=[0, 15], density=True, alpha=0.3, color='red', label='Geant')

    if fakeWidth is not None:
        plt.hist(fakeWidth, bins=50, range=[0, 15], density=True, alpha=0.3, color='blue', label='GAN')
    plt.legend(loc='best')
    plt.xlabel(('Longitudual' if orto else 'Transverse') + ' cluster width [cm]')
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


def plotEnergies(ecal, fake=None) :
    maxE = max(ecal)
    plt.hist(ecal, 100, range=[0, maxE], log=True, color='red', alpha=0.3)
    legend = ['Geant']

    if fake is not None:
        plt.hist(fake, 100, range=[0, maxE], log=True, color='blue', alpha=0.3)
        legend.append('GAN')

    plt.legend(legend)
    plt.xlabel('Energy')
    plt.ylabel('Arbitrary Units')