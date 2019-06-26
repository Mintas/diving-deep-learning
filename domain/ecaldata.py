import numpy as np
from os.path import dirname
import torchvision.transforms as tvtf


class EcalData :
    def __init__(self, response, momentum, point, title='') -> None:
        self.response = response
        self.momentum = momentum
        self.point = point
        self.title = title


def parseEcalData(filename):
    ecal = np.load(dirname(dirname(__file__)) + '/resources/ecaldata/' + filename + '.npz')
    ecal.keys()
    return dictToEcalData(ecal)


def dictToEcalData(ecal):
    real_imgs = ecal['EnergyDeposit']
    real_p = ecal['ParticleMomentum']
    real_point = ecal['ParticlePoint']
    title = 'EnergyDeposit  shape: ' + str(real_imgs.shape) + \
            '\n min: ' + str(real_imgs.min()) + '\t max: ' + str(real_imgs.max()) + \
            '\n first particle Momentum :  ' + str(real_p[0]) + \
            '\n Point :' + str(real_point[0]) + \
            '\n particle type is : ' + str(ecal['ParticlePDG'][0])
    return EcalData(real_imgs, real_p, real_point, title)

def resizeResponses(data, size) :
    resize = tvtf.Compose([
        tvtf.ToPILImage(),
        tvtf.Resize(size),
        tvtf.ToTensor()
    ])
    response = data.response if isinstance(data, EcalData) else data
    responseResized = np.array([resize(np.float32(img))[0].numpy() for img in response])
    return EcalData(responseResized, data.momentum, data.point, data.title + '\n Resized to ' + str(size) + 'x' + str(size)) if isinstance(data, EcalData) else responseResized