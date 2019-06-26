import torch
import os


EXTENSION = '.pth'

class IOGANConst:
    FixedNoise = 'fixedNoise'
    D = 'D_state'
    G = 'G_state'
    Dopt = 'Dopt'
    Gopt = 'Gopt'
    Epoch = 'epoch'
    DLoss = 'DLoss'
    GLoss = 'GLoss'


def isFilePresent(PATH) :
    return os.path.isfile(PATH + EXTENSION)

def saveGAN(epoch, fixedNoise, D, Dopt, Dloss, G, Gopt, Gloss, PATH):
    print('!!!!!! SAVING MODELS DONT BREAK EXECUTION !!!!!!')
    torch.save({
            IOGANConst.FixedNoise : fixedNoise,
            IOGANConst.D: D.state_dict(),
            IOGANConst.G: G.state_dict(),
            IOGANConst.Dopt: Dopt.state_dict(),
            IOGANConst.Gopt: Gopt.state_dict(),
            IOGANConst.Epoch: epoch,
            IOGANConst.DLoss: Dloss,
            IOGANConst.GLoss: Gloss
            }, PATH + EXTENSION)
    print('!!!!!! Models have been saved successfully !!!!!')


def loadGAN(D, Dopt, G, Gopt, PATH):
    checkpoint = torch.load(PATH + EXTENSION)

    D.load_state_dict(checkpoint[IOGANConst.D])
    G.load_state_dict(checkpoint[IOGANConst.G])
    Dopt.load_state_dict(checkpoint[IOGANConst.Dopt])
    Gopt.load_state_dict(checkpoint[IOGANConst.Gopt])

    DL = checkpoint[IOGANConst.DLoss]
    GL = checkpoint[IOGANConst.GLoss]
    return checkpoint[IOGANConst.Epoch], DL, GL, checkpoint[IOGANConst.FixedNoise]


def loadGANs(D, G, PATH):
    checkpoint = torch.load(PATH + EXTENSION)

    D.load_state_dict(checkpoint[IOGANConst.D])
    G.load_state_dict(checkpoint[IOGANConst.G])