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


#runSomeReview()







