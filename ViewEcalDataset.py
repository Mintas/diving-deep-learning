import numpy as np
from os.path import dirname
from timeit import default_timer as timer
from analytics import analytic_funcs as AF
from analytics import optimized_analytic_funcs as OAF
import torchvision.transforms as tvtf
import matplotlib.pyplot as plt
import torch

import domain.ecaldata as ED
from serialization import iogan
from collections import OrderedDict

# EnergyDeposit = ecal['EnergyDeposit']
# ParticlePoint = ecal['ParticlePoint']
# ParticleMomentum = ecal['ParticleMomentum']
# ParticlePDG=ecal['ParticlePDG']  #here we got only vector with constant 22

dataset = 'caloGAN_v3_case4_2K' #'caloGAN_v3_case2_50K'#'caloGAN_v3_case4_2K'
data = ED.parseEcalData(dataset)

size = 8

res = OrderedDict()
for i in range(0, data.count()):
    kv = data.getKeyValued(i)
    res.setdefault(kv[0], []).append(kv[1])

# resize = tvtf.Compose([
#     tvtf.ToPILImage(),
#     tvtf.Resize(size),
#     tvtf.ToTensor()
#     ])
# responseResized = np.array([resize(np.float32(img))[0].numpy() for img in data.response])
# dataResized = ED.EcalData(responseResized, data.momentum, data.point, data.title)

dataResized = ED.resizeResponses(data, size)

# fig = plt.figure(figsize=(10, 10))
# AF.plotResponses(dataResized)
# fig.show()

outputName = dataset + '_resizedto_' + str(size)
if not iogan.isFilePresent(outputName):
    sizedAnalytic = OAF.optimized_analityc(dataResized, size)
    torch.save(sizedAnalytic, outputName + iogan.EXTENSION)
else:
    sizedAnalytic = torch.load(outputName + iogan.EXTENSION)

OAF.runAnalytics('/resources/resized/' + outputName, dataResized, ecalStats=sizedAnalytic)

#start = timer()
#OAF.runAnalytics(dataset, ecalData=data)  # , fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
#end = timer()
#print(end - start)

#start = timer()
#AF.runAnalytics(dataset, ecalData=data)  # , fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
#end = timer()
#print(end - start)
