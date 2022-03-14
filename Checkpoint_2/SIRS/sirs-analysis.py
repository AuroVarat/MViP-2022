from cv2 import sqrt
from matplotlib import pyplot as plt
from main import main
import numpy as np
from tqdm import tqdm
import itertools
#n = 1000

x = np.linspace(0,1,21)
print(x)
# XX,YY = np.meshgrid(x,x)
n = len(x)
#mesh = np.array(np.meshgrid(x,x)).T.reshape(n,-1,2)


ai = np.empty((n,5))
#vi = np.empty((n,2))    

for i in tqdm(range(n)):
    for j in range(n):
        pi,pj = mesh[i,j]
        ai[i,j], vi[i,j]= main(p1=pi,p2=0.5,p3=pj)
        
for i in tqdm(range(n)):
        for l in range(5):
        # pi,pj = mesh[i,j]
                ai[i,l]= main(p1=0.5,p2=0.5,p3=0.5,fim=x[i])
    

aib = np.mean(ai,axis=1)
err = np.std(ai,axis=1)/np.sqrt(5)
# pyplot.hist(ai,bins = 70,range=(0,np.nanmax(ai))) 
# pyplot.show()     
#np.savetxt("Checkpoint_2/SIRS/output-mean-reduced.dat", ai)
np.savetxt("Checkpoint_2/SIRS/output-mean-immunized-wavs.dat", np.c_[aib,err])
#ai = np.genfromtxt("Checkpoint_2/SIRS/output-mean.dat")

plt.errorbar(x,ai,yerr=err,fmt='.-')
plt.show()
# plt.contourf(XX,YY,ai)
# plt.colorbar()
# plt.show()

# plt.contourf(XX,YY,vi)
# plt.colorbar()
# plt.show()