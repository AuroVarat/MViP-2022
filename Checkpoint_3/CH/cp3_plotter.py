from matplotlib import pyplot as plt

import numpy as np



x = np.linspace(0,int(1e+6),int(1e+4)+1)
ai = np.genfromtxt("Checkpoint_3/CH/data/output_bubble_final.dat")

plt.plot(ai)
plt.show()