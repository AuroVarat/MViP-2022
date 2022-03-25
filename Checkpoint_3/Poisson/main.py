from os import system
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pb
plt.rcParams["figure.figsize"] = [7.00, 7.00]
plt.rcParams["figure.autolayout"] = True

np.random.seed(10)
N = 50
c_N = N//2
nstep = int(1e+4)+1
dx = 1.
e0 = 1.
s = slice(1,N+1)
r1by6 = 1/6
#calc
system = np.zeros((N+2,N+2,N+2))

rho = np.zeros((N+2,N+2,N+2))
rho[c_N,c_N,c_N] = 1
#rho[c_N+2,c_N+2,c_N+2] = -1
#rho[c_N-5:c_N+5,c_N-5:c_N+5,c_N-5:c_N+5] = np.random.choice([1,0.5],(10,10,10))


    
def update():
    global system
    
    prev_system = system.copy()
    
    system = r1by6*(    
                        np.pad
                        ( 
                            np.roll( prev_system,1,axis=0)[s,s,s]+ 
                            np.roll(prev_system,-1,axis=0)[s,s,s] + 
                            np.roll(prev_system,1,axis=1)[s,s,s] + 
                            np.roll(prev_system,-1,axis=1)[s,s,s] +
                            np.roll(prev_system,1,axis=2)[s,s,s]+ 
                            np.roll(prev_system,-1,axis=2)[s,s,s],pad_width=1
                        )+ rho
                        
                    )
 

    
    return np.sum(np.absolute(system-prev_system))



for sweep in pb(range(nstep)):

    if np.isclose(update(),0,atol=1e-3):
        print(sweep)
        break 



plt.imshow(system[0:N+2,0:N+2,c_N],interpolation='nearest')
plt.colorbar()
plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# system = system[0:c_N,0:c_N,0:N]
# x,y,z = (system > 0.001).nonzero()
# k = system.flatten()

# k=k[~np.isclose(k,0,atol=1e-3)]
# print(k[0])
# p = ax.scatter(x,y,z,c=k,alpha=1)
# fig.colorbar(p, ax = ax, shrink = 0.5, aspect = 5)
# plt.show()
#np.savetxt("Checkpoint_3/Poisson/data/output.dat",system)