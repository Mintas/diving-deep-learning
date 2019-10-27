import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

datasetName = 'caloGAN_v4_case2_50K'
ecalData = np.load('/Users/mintas/PycharmProjects/untitled1/resources/ecaldata/%s.npz' % datasetName)
ed = ecalData['EnergyDeposit']

def transform(ed) : #here we transform (50k, 30x30) dataset to (30x30) datasets of 50k size, (i,j) - pixel index
    return np.transpose(ed, (1, 2, 0))

ed = transform(ed)


