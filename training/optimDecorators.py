from torch import optim as optim


def optAdam(netParameters, hyperParams):
    return optim.Adam(netParameters, lr=hyperParams.lr, betas=(hyperParams.beta, 0.999))


def optRMSProp(netParameters, hyperParams):
    return optim.RMSprop(netParameters, lr=hyperParams.lr)


def optSGD(netParameters, hyperParams):
    return optim.SGD(netParameters, lr=hyperParams.lr)