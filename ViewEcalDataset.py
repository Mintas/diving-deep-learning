import numpy as np

#file contains 50k observations
ecal = np.load('resources/electrons_1_100_5D_v3_50K.npz')
ecal.keys()

# EnergyDeposit = ecal['EnergyDeposit']
# ParticlePoint = ecal['ParticlePoint']
# ParticleMomentum = ecal['ParticleMomentum']
# ParticlePDG=ecal['ParticlePDG']  #here we got only vector with constant 22
