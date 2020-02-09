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

#case4 =  [ 1.75761939, -2.4006713 , 16.63603172] , 16.9
#case3 = [-1.59858634  0.80942614 29.04478182], 29.1
#case2 = [0.15657315 0.14722511 2.4455745 ], 2.455

#dataset = 'caloGAN_v4_case2_50K' #'caloGAN_v3_case2_50K'#'caloGAN_v3_case4_2K'
#ecal = torch.load('/Users/mintas/Downloads/caloGAN_batch100_1of2_cLinearDeep_WGAN.pth', map_location='cpu')

dataset = 'caloGAN_batch100_1of2' #'caloGAN_v3_case2_50K'#'caloGAN_v3_case4_2K'
data = ED.parseEcalData(dataset)
data.response = np.squeeze(data.response, -1)
ecalStatsPath = '/Users/mintas/' + dataset + '_stats' + '.pth'
torch.save(OAF.optimized_analityc(data, 30), ecalStatsPath)


# size = 8
#
# res = OrderedDict()
# for i in range(0, data.count()):
#     kv = data.getKeyValued(i)
#     res.setdefault(kv[0], []).append(kv[1])
for i in range(1,100) :
    print(data.momentum[i*10], data.point[i*10])

# resize = tvtf.Compose([
#     tvtf.ToPILImage(),
#     tvtf.Resize(size),
#     tvtf.ToTensor()
#     ])
# responseResized = np.array([resize(np.float32(img))[0].numpy() for img in data.response])
# dataResized = ED.EcalData(responseResized, data.momentum, data.point, data.title)

# dataResized = ED.resizeResponses(data, size)

# fig = plt.figure(figsize=(10, 10))
# AF.plotResponses(dataResized)
# fig.show()

# outputName = dataset + '_resizedto_' + str(size)
# if not iogan.isFilePresent(outputName):
#     sizedAnalytic = OAF.optimized_analityc(dataResized, size)
#     torch.save(sizedAnalytic, outputName + iogan.EXTENSION)
# else:
#     sizedAnalytic = torch.load(outputName + iogan.EXTENSION)
#
# OAF.runAnalytics('/resources/resized/' + outputName, dataResized, ecalStats=sizedAnalytic)

#start = timer()
#OAF.runAnalytics(dataset, ecalData=data)  # , fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
#end = timer()
#print(end - start)

#start = timer()
#AF.runAnalytics(dataset, ecalData=data)  # , fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
#end = timer()
#print(end - start)
