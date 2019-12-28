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
    if (withEnergy): arrayOfTensors.append(energy(momentums, True))
    return torch.cat(arrayOfTensors, 1)  # 50k * (x,y,px,py,pz,E0)

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
    datasetName = 'caloGAN_v4_case0_50K'
    ecalData = ed.parseEcalData(datasetName)  # '/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' %
    energyDeposites = matrixOfEnergyDepositeSamples(ecalData)

    conditions = buildConditionsBySample(ecalData, True)

    mc = torch.mean(conditions, 0, True)
    ms = torch.std(conditions, 0, True)
    conditions = standardizeConditions(conditions)  # Zero-mean, Unit-variance


    def scatterHtml(z, zTitle, markers, filename):
        fig1 = go.Scatter3d(x=conditions[:, 0],
                            y=conditions[:, 1],
                            z=z,
                            marker=dict(color=markers,
                                        opacity=1,
                                        reversescale=True,
                                        colorscale='Blues',
                                        size=5),
                            line=dict(width=0.02),
                            mode='markers')
        # Создаём layout
        mylayout = go.Layout(scene=dict(xaxis=dict(title="Xcoord"),
                                        yaxis=dict(title="Ycoord"),
                                        zaxis=dict(title=zTitle)), )
        # Строим диаграмму и сохраняем HTML
        plotly.offline.plot({"data": [fig1],
                        "layout": mylayout},
                            auto_open=True,
                            filename=(filename))

    # Создаём figure
    i, j = 15, 15
    # 'Deposit'
    #scatterHtml(energyDeposites[i, j], "E0", conditions[:, 5], "StandardizedE0byDeposit_{}_{}.html".format(i, j))
    energies = knnEnergies(conditions, energyDeposites[i, j], 1000)
    scatterHtml(conditions[:, 5], "E0", torch.tensor(energies), "E0_KNN_{}_{}.html".format(i, j))


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







