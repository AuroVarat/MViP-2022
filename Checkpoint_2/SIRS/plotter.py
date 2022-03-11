from matplotlib import pyplot as plt
from main import main
import numpy as np
from tqdm import tqdm




x = np.linspace(0,1,21)
XX,YY = np.meshgrid(x,x)
ai = np.genfromtxt("Checkpoint_2/SIRS/output-mean.dat")
vi = np.genfromtxt("Checkpoint_2/SIRS/output-var.dat")

plt.contourf(XX,YY,ai)
#plt.imshow(ai,extent=[0,1,0,1],origin='lower')
plt.colorbar()
plt.show()

#plt.contourf(XX,YY,vi)
plt.imshow(vi,extent=[0,1,0,1],origin='lower')
plt.colorbar()
plt.show()

x = np.linspace(0.2,0.5,30)
ai_red = np.genfromtxt("Checkpoint_2/SIRS/output-var-reduced.dat")
plt.errorbar(x,ai_red[:,0],yerr=ai_red[:,1],ecolor='red',fmt='--',alpha=0.5,capsize=3.0)
plt.show()

x = np.linspace(0,1,21)
ai_immune = np.genfromtxt("Checkpoint_2/SIRS/output-mean-immunized-dyeq.dat")
plt.errorbar(x,ai_immune[:,0],yerr=ai_immune[:,1],ecolor='red',fmt='--',alpha=0.5,capsize=3.0)
plt.show()