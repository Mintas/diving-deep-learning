import numpy as np
from os.path import dirname
import torchvision.transforms as tvtf


class EcalData :
    def __init__(self, response, momentum, point, title='') -> None:
        self.response = response
        self.momentum = momentum
        self.point = point
        self.title = title

    def count(self):
        return self.response.shape[0]

    def get(self, i):
        return (self.response[i], self.momentum[i], self.point[i])

    def getKeyValued(self, i):
        return (EcalPoint(self.momentum[i], self.point[i]), self.response[i])

class EcalPoint :
    def __init__(self, momentum, point) -> None:
        self.momentum = momentum
        self.point = point

    def __eq__(self, other):
        return (self.roundM(self.momentum) == self.roundM(other.momentum)).all() \
               and (self.roundM(self.point) == self.roundM(other.point)).all()

    def __hash__(self):
        return hash((*self.roundM(self.momentum), *self.roundM(self.point)))

    def roundM(self, m):
        return np.around(m, decimals=0)


def parseEcalData(filename, fulldata=True):
    ecal = np.load(dirname(dirname(__file__)) + '/resources/ecaldata/' + filename + '.npz')
    ecal.keys()
    return dictToEcalData(ecal, fulldata)


def dictToEcalData(ecal, fulldata=True):
    real_imgs = ecal['EnergyDeposit']
    real_p = ecal['ParticleMomentum']
    real_point = ecal['ParticlePoint']
    rounded = lambda num : str(np.round(num, 3))
    title = 'EnergyDeposit  shape: ' + str(real_imgs.shape) + \
            '\n min: ' + rounded(real_imgs.min()) + '; max: ' + rounded(real_imgs.max()) + \
            '\n first particle Momentum :  ' + rounded(real_p[0]) + \
            '\n Point :' + rounded(real_point[0]) + \
            ('\n particle type is : ' + str(ecal['ParticlePDG'][0])) if fulldata else ''
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

def averageDataset(datasetName, avgCnt=100):
    ecalData = parseEcalData(datasetName)  # '/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' %
    print('read dataset : done')
    # ecalData.response = np.mean(ecalData.response, -1)
    ecalData.response = np.squeeze(ecalData.response, -1)
    print('squeezed !')
    splitted = np.split(ecalData.response,
                        int(ecalData.response.shape[0] / avgCnt))  # split by 100 samples (they must have equal conditions)
    print('splitted !')
    averaged = [np.mean(split, axis=0, keepdims=False) for split in splitted]
    print('averaged !')
    ecalData.response = np.asarray(averaged)
    ecalData.momentum = ecalData.momentum[0::avgCnt]
    ecalData.point = ecalData.point[0::avgCnt]
    np.savez_compressed('averaged' + datasetName,
                        EnergyDeposit=ecalData.response,  # 30х30 матрица откликов,
                        ParticlePoint=ecalData.point,  # компоненты координат, трехмерные
                        ParticleMomentum=ecalData.momentum)  # компоненты импульса частицы, трехмерные

def joinDatasets(ds1, ds2):
    ed1 = parseEcalData(ds1)  # '/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' %
    ed2 = parseEcalData(ds2)

    title = "joined_{}_and_{}".format(ds1, ds2)
    ecalData = EcalData(np.array([]), np.array([]), np.array([]), title)
    ecalData.response = np.concatenate((ed1.response, ed2.response))
    ecalData.momentum = np.concatenate((ed1.momentum, ed2.momentum))
    ecalData.point = np.concatenate((ed1.point, ed2.point))

    np.savez_compressed(title,
                        EnergyDeposit=ecalData.response,  # 30х30 матрица откликов,
                        ParticlePoint=ecalData.point,  # компоненты координат, трехмерные
                        ParticleMomentum=ecalData.momentum)
    return ecalData
