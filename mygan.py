import torch.nn as nn


class GANS :
    GAN = 'GAN'
    WGAN = 'WGAN'
    CRAMER = 'CRAMER'
    ECRAMER = 'ECRAMER'


def weights_init(m):     # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def num_flat_features(x):
    size = x.size()
    num_features = 1
    for s in size:
        num_features *= s
    return num_features