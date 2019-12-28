import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
from scipy.interpolate import interp2d
import math

#%matplotlib inline

CELLSIZE=2.

real_data = np.load('/Users/mintas/PycharmProjects/untitled/resources/ecaldata/caloGAN_v4_case3_10K.npz')

