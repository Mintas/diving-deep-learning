import torch
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio

import regression.caloreg as caloreg
import domain.ecaldata as ed


torch.manual_seed(1)  # reproducible

BATCH_SIZE = 1000
EPOCH = 100
i,j=15,15


datasetName = 'caloGAN_v4_case0_50K'
ecalData = ed.parseEcalData(datasetName)  # '/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' %
energyDeposites = caloreg.matrixOfEnergyDepositeSamples(ecalData)
conditions = caloreg.buildConditionsBySample(ecalData, withEnergy=False)
conditions = caloreg.standardizeConditions(conditions) #Zero-mean, Unit-variance

responses = energyDeposites[i, j]
print('gonna KNN responses')
responses = torch.tensor(caloreg.knnEnergies(conditions, responses, 1000)) #replace responses with KNN responses, clustering
print('KNN finished')

torch_dataset = Data.TensorDataset(conditions, responses)

# another way to define a network
net = torch.nn.Sequential(
    torch.nn.Linear(conditions.size(1), 100),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(100, 70),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(70, 30),
    torch.nn.Tanh(),
    torch.nn.Linear(30, 1),
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss



loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, )

my_images = []
fig, ax = plt.subplots(figsize=(16, 10))


def plotAndShowLearningProcess(fig, ax, conditions, responses, regressed, finish, conditionId=1):
    # plot and show learning process
    plt.cla()
    ax.set_title('Regression Analysis at pixel i={},j={}'.format(i,j), fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    # ax.set_xlim(-11.0, 11.0)
    # ax.set_ylim(-0.1, 11000)
    ax.scatter(conditions[:, conditionId].numpy(), responses.numpy(), color="blue", alpha=0.2)
    ax.scatter(conditions[:, conditionId].numpy(), regressed.numpy(), color='green', alpha=0.5)
    finish()


def finishScattering():
    ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
            fontdict={'size': 24, 'color': 'red'})
    ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color': 'red'})
    # Used to return the plot as an image array
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    my_images.append(image)


# start training
for epoch in range(EPOCH):
    if epoch % 1 == 0:
        print("Epoch : {}".format(epoch))
    for step, (b_x, b_y) in enumerate(loader):  # for each training step
        prediction = net(b_x)  # input x and predict based on x

        loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if epoch % 10==0 and step == 1:
            with torch.no_grad():
                plotAndShowLearningProcess(fig, ax, b_x.data, b_y.data, prediction.data, finishScattering)

# save images as a gif
imageio.mimsave('./reg_{}-{}_c{}.gif'.format(i,j,1), my_images, fps=12)

with torch.no_grad():
    prediction = net(conditions)  # input x and predict based on x

    for c in range(5) :
        fig, ax = plt.subplots(figsize=(16, 10))
        def saveAndSHowFig():
            plt.savefig('reg_{}-{}_c{}.png'.format(i,j,c))
            plt.show()
        plotAndShowLearningProcess(fig, ax, conditions, responses, prediction, saveAndSHowFig, c)
