
import matplotlib
matplotlib.use('TKAgg')
import time
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

J=1.0
nstep=10000
np.random.seed(10)
#input

# if(len(sys.argv) != 3):
#     print ("Usage python ising.animation.py N T")
#     sys.exit()

# lx=int(sys.argv[1]) 
# ly=lx 
# kT=float(sys.argv[2]) 
lx = 50
ly = lx
kT = 1

def mic(vector,box_size):
    """
        Finds the image of the points with MIC

        :param vector: Any kind of vectorized element in list or on its own
        :param box_size: size of the simulation box

        :return MIC enforced vector image: vector
        
    """
    
    return np.subtract(np.mod(np.add(vector,np.multiply(box_size,0.5)),box_size),np.multiply(box_size,0.5))

def total_energy(N):
       
    spin = np.random.choice([-1,1],[N,N])
    
    pos_list = np.array(np.meshgrid(np.arange(0,N), np.arange(0,N))).T.reshape(-1,2)
    separation_matrix = np.ones((N**2,N**2,2))*pos_list
    separation_matrix = mic(np.swapaxes((pos_list-np.swapaxes(separation_matrix, 0, 1)), 0, 1),N)
    separation_matrix[np.tril_indices(n=N**2, k=0)]=0
    separation_matrix = np.linalg.norm(separation_matrix,axis=2)
    separation_matrix = np.where(separation_matrix <= 1,separation_matrix,0)


    spin_list = spin.reshape(1,N**2)
    spin_matrix = np.ones((N**2,N**2))*spin_list
    spin_neighbor_matrix = np.multiply(separation_matrix,spin_list)
    energy_matrix = -1*spin_neighbor_matrix*spin_matrix
    energy = np.sum(energy_matrix)
    return spin,energy,spin_neighbor_matrix

def pair_energy_calc(Si,Sj):
    return -1*Si*Sj

#initialise spins randomly
spin,E,Sj_matrix = total_energy(ly)


#calculate energy

# E = 0

# for i in range(lx):
#     for j in range(ly):
#         e = 0
#         e+= -pair_energy_calc(spin[i,j],spin[(i+1)%lx,j])
#         e+= -pair_energy_calc(spin[i,j],spin[(i-1)%lx,j])
#         e+= -pair_energy_calc(spin[i,j],spin[i,(j+1)%ly])
#         e+= -pair_energy_calc(spin[i,j],spin[(i+1)%lx,(j-1)%ly])
#         E+=e
# E = E/2

print(E)
fig = plt.figure()
im=plt.imshow(spin, animated=True)

#update loop here - for Glauber dynamics

t1 = time.time()
for n in range(nstep):
    itrial, jtrial = np.random.randint(0,50, size=2)
    
    for i in range(lx):
        for j in range(ly):

#select spin randomly
            itrial=np.random.randint(0,lx)
            jtrial=np.random.randint(0,ly)
            spin_new=-spin[itrial,jtrial]

#compute delta E eg via function (account for periodic BC)
            e = 0
            e+= pair_energy_calc(spin_new,spin[(itrial+1)%lx,jtrial])
            e+= pair_energy_calc(spin_new,spin[(itrial-1)%lx,jtrial])
            e+= pair_energy_calc(spin_new,spin[itrial,(jtrial+1)%ly])
            e+= pair_energy_calc(spin_new,spin[itrial,(jtrial-1)%ly])
            e = 2*e
            if e <= 0:
                spin[itrial,jtrial] = spin_new
            else:
                prob= np.exp(-e/kT)
                spin[itrial,jtrial] *= np.random.choice([-1,1],p=[prob,1-prob])


#perform metropolis test
                
#occasionally plot or update measurements, eg every 10 sweeps
    if(n%10==0): 
#       update measurements
#       dump output
        
        f=open('spins.dat','w')
        for i in range(lx):
            for j in range(ly):
                f.write('%d %d %lf\n'%(i,j,spin[i,j]))
        f.close()
#       show animation
        plt.cla()
        plt.title(n)
        im=plt.imshow(spin, animated=True)
        plt.draw()
        plt.pause(0.0001)
       
t2 = time.time()

print(t2-t1)

