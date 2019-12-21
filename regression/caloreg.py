import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
import torch

datasetName = 'caloGAN_v4_case0_50K'
ecalData = np.load('/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' % datasetName)
ed = ecalData['EnergyDeposit']

def transform(ed) : #here we transform (50k, 30x30) dataset to (30x30) datasets of 50k size, (i,j) - pixel index
    return np.transpose(ed, (1, 2, 0))

ed = transform(ed)

toFloatTensor = lambda key: torch.from_numpy(ecalData[key]).float()
condition = torch.cat([toFloatTensor('ParticlePoint')[..., :2], toFloatTensor('ParticleMomentum')], 1)

mc = torch.mean(condition, 0, True)
ms = torch.std(condition, 0, True)

condition = (condition - mc) / ms



