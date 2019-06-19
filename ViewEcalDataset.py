import numpy as np
from os.path import dirname
from timeit import default_timer as timer
from analytics import analytic_funcs as AF
from analytics import optimized_analytic_funcs as OAF
import domain.ecaldata as ED

# EnergyDeposit = ecal['EnergyDeposit']
# ParticlePoint = ecal['ParticlePoint']
# ParticleMomentum = ecal['ParticleMomentum']
# ParticlePDG=ecal['ParticlePDG']  #here we got only vector with constant 22

dataset = 'caloGAN_v3_case2_50K'#'caloGAN_v3_case4_2K'
data = ED.parseEcalData(dataset)

start = timer()
OAF.runAnalytics(dataset, ecalData=data)  # , fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
end = timer()
print(end - start)

start = timer()
AF.runAnalytics(dataset, ecalData=data)  # , fakeData=ED.parseEcalData('caloGAN_v3_case5_2K'))
end = timer()
print(end - start)