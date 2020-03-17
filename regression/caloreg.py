import numpy as np
import torch
import domain.ecaldata as ed
import plotly
import plotly.graph_objs as go


def toFloatTensor(nparr):
    return torch.from_numpy(nparr).float()


def matrixOfEnergyDepositeSamples(ecaldata): #here we transform (50k, 30x30) dataset to (30x30) datasets of 50k size, (i,j) - pixel index
    return toFloatTensor(np.transpose(ecaldata.response, (1, 2, 0)))


energy = lambda momentum, keepdim: torch.norm(momentum, dim=-1, keepdim=keepdim)


def buildConditionsBySample(ecaldata, withEnergy):
    momentums = toFloatTensor(ecaldata.momentum)
    arrayOfTensors = [toFloatTensor(ecaldata.point)[..., :2], momentums]
    if (withEnergy):
        energies = energy(momentums, True)
        arrayOfTensors.append(energies)
        arrayOfTensors.append(momentums/energies) #add angles px/E0, py/E0, pz/E0

    return torch.cat(arrayOfTensors, 1)  # 50k * (x,y,px,py,pz,E0, ax, ay, az)

def standardizeConditions(conditions):
    mc = torch.mean(conditions, 0, True)
    ms = torch.std(conditions, 0, True)
    return (conditions - mc) / ms

def knnEnergies(conditions, nrgs, nn=50):
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=nn)
    neigh.fit(conditions)

    #nrgs = energyDeposites[i, j]
    ii=0
    def knnEnergy(i):
        iiNeighs = neigh.kneighbors([conditions[i].numpy()])
        return torch.mean(nrgs[iiNeighs[1]])

    return [knnEnergy(i) for i in range(nrgs.size(0))]


def runSomeReview():
    datasetName = 'caloGAN_batch100_1of2'
    datasetName = 'averaged_' + datasetName
    ecalData = ed.parseEcalData(datasetName, False)
    #ed.averageDataset(datasetName)

    energyDeposites = matrixOfEnergyDepositeSamples(ecalData)

    # 50k * (x,y,px,py,pz,E0, ax, ay, az)
    conditions = buildConditionsBySample(ecalData, True)

    mc = torch.mean(conditions, 0, True)
    ms = torch.std(conditions, 0, True)
    #conditions = standardizeConditions(conditions)  # Zero-mean, Unit-variance

    print('conditions and responses : done')

    #scatter SUM for (i,j) on main diag, DIF for (i,j) on countermain; use x*(14.5 - i) + y*(14.5 - j) to weigted plot
    def scatterHtml(z, zTitle, markers, filename):
        fig1 = go.Scatter3d(x=conditions[:, 0] * (14.5-7) + conditions[:, 1] * (14.5-15), #x=conditions[:, 6],#x=conditions[:, 0],
                            y=conditions[:, 6] * (14.5-7) + conditions[:, 7] * (14.5-15),#y=conditions[:, 1],
                            z=z,
                            marker=dict(color=markers,
                                        opacity=1,
                                        reversescale=True,
                                        colorscale='Blues',
                                        colorbar=dict(
                                            title="Colorbar"
                                        ),
                                        size=5),
                            line=dict(width=0.02),
                            mode='markers')
        # Создаём layout
        mylayout = go.Layout(scene=dict(xaxis=dict(title="x - y"),#"Xcoord"),
                                        yaxis=dict(title="(Px - Py)/E0"),#"Ycoord"),
                                        zaxis=dict(title=zTitle)), )
        # Строим диаграмму и сохраняем HTML
        plotly.offline.plot({"data": [fig1],
                        "layout": mylayout},
                            auto_open=True,
                            filename=(filename))

    # Создаём figure
    i, j = 7, 15
    # 'Deposit'
    #scatterHtml(energyDeposites[i, j], "E0", conditions[:, 5], "StandardizedE0byDeposit_{}_{}.html".format(i, j))
    #energies = knnEnergies(conditions, energyDeposites[i, j], 100)
    energies = energyDeposites[i, j]
    scatterHtml(conditions[:, 5], "E0", torch.tensor(energies), "{}_2difxy_difPxPy_E0_by100_{}_{}.html".format(datasetName,i, j))
    #scatterHtml(conditions[:, 5], "E0", torch.tensor(energies), "{}_PxPyE0_by100_{}_{}.html".format(datasetName,i, j))


import matplotlib.pyplot as plt



def tryKNNRegression():
    datasetName = 'caloGAN_v4_case0_50K'
    ecalData = ed.parseEcalData(datasetName)  # '/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' %
    energyDeposites = matrixOfEnergyDepositeSamples(ecalData)

    conditions = buildConditionsBySample(ecalData, True)
    conditions = standardizeConditions(conditions)  # Zero-mean, Unit-variance

    i,j = 15,15
    responses = energyDeposites[i,j]
    #responses = knnEnergies(conditions, responses, 1000)

    print('prepared, gonna fit!')

    from sklearn.neighbors import KNeighborsRegressor
    n_neighbors = 0
    distance = 'distance'
    knn = KNeighborsRegressor(n_neighbors, weights=distance)

    knn.fit(conditions[:10000], responses[:10000])
    print('fitted, gonna predict!')


    predicted = knn.predict(conditions)

    def plotAndShowLearningProcess(fig, ax, conditions, responses, regressed, finish, conditionId=1):
        # plot and show learning process
        plt.cla()
        ax.set_title('Regression Analysis at pixel i={},j={}, NN={}, distance={}'.format(i, j, n_neighbors, distance), fontsize=35)
        ax.set_xlabel('Independent variable', fontsize=24)
        ax.set_ylabel('Dependent variable', fontsize=24)
        # ax.set_xlim(-11.0, 11.0)
        # ax.set_ylim(-0.1, 11000)
        ax.scatter(conditions[10000:11000, conditionId].numpy(), np.array(responses[10000:11000]), color="blue", alpha=0.2)
        ax.scatter(conditions[10000:11000, conditionId].numpy(), np.array(regressed[10000:11000]), color='red', alpha=0.3)
        finish()

    for c in range(5) :
        fig, ax = plt.subplots(figsize=(16, 10))
        def saveAndSHowFig():
            plt.savefig('KNNreg_{}-{}_c{}.png'.format(i,j,c))
            plt.show()
        plotAndShowLearningProcess(fig, ax, conditions, responses, predicted, saveAndSHowFig, c)


#tryKNNRegression()
#runSomeReview()


def runPointedHistograms():
    #datasetName = 'caloGAN_v4_case2_50K'
    #indexes = [[15,15,280], [13,13,30], [11,11,7.5], [9,9,5]] #i,j, upperEnergyThreshold

    datasetName = 'caloGAN_v4_case3_10K'
    indexes = [[15, 15, 1100], [13, 13, 65], [11, 11, 12.5], [9, 9, 10]]  # i,j, upperEnergyThreshold

    ecalData = ed.parseEcalData(datasetName)  # '/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' %
    energyDeposites = matrixOfEnergyDepositeSamples(ecalData)

    from os.path import dirname
    import plots.plotUi as PUI
    import analytics.plotAnalytics as pa
    import seaborn as sns

    outputfile = dirname(dirname(__file__)) + '/resources/points/' + datasetName + '_PointDistr'
    plotUi = PUI.PDFPlotUi(outputfile)  # PUI.ShowPlotUi()


    def plotPointEnergyDistr():
        pos = 1
        histType = 'stepfilled'
        for ind in indexes:
            pointNrgs = energyDeposites[ind[0], ind[1]].numpy()
            pointNrgs = pointNrgs[np.where(pointNrgs < ind[2])]

            plt.subplot(4, 2, pos)
            sns.distplot(pointNrgs, hist=True, kde=True, bins=100,
                         hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 1})
            pos = pos + 1

            plt.subplot(4, 2, pos)
            sns.distplot(pointNrgs, hist=True, kde=False, bins=100,
                         hist_kws={'edgecolor': 'black', 'log':True}, kde_kws={'linewidth': 1})
            pos = pos + 1


    plotUi.toView(plotPointEnergyDistr)
    plotUi.close()


#runPointedHistograms()


def doJoinDataSets():
    datasetName = 'caloGAN_v4_case3_10K'
    datasetName = 'averaged_' + datasetName

    ecalData = ed.joinDatasets('joined_v4_3_10K_4_10K', '2_50K_5_10K')
    print('yay')
    #ed.averageDataset(datasetName)

#doJoinDataSets()

