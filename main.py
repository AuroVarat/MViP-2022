
import posixpath
from xmlrpc.client import boolean
import matplotlib
from sklearn import neighbors
matplotlib.use('TKAgg')
import time
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cProfile
from tqdm import tqdm
from scipy import sparse
import cProfile


J=1.0
nstep=10000
np.random.seed(10)
#terminal input



N = int(input("System Size: "))
kT = int(input("Temperature: "))
dyanmics = str(input("Dynamics Algorithm(g or k): "))
animation = bool(int(input("Animation: ")))
analysis_mode = bool(int(input("Analysis Mode: ")))

if analysis_mode:
    final_temp =  int(input("Final Temperature: "))
    step_size = float(input("Step Size: "))
else:
    final_temp = kT
    step_size = 1
#initialise spins randomly
spin = np.random.choice([-1,1],[N,N])

def mic(vector,box_size):
    """
        Finds the image of the points with MIC

        :param vector: Any kind of vectorized element in list or on its own
        :param box_size: size of the simulation box

        :return MIC enforced vector image: vector
        
    """
    return np.subtract(np.mod(np.add(vector,np.multiply(box_size,0.5)),box_size),np.multiply(box_size,0.5))

def total_energy(spin,N):
    spin_list = spin.reshape(1,N**2)

    pos_list = np.array(np.meshgrid(np.arange(0,N), np.arange(0,N))).T.reshape(-1,2)
    separation_matrix = np.ones((N**2,N**2,2))*pos_list
    separation_matrix = mic(np.swapaxes((pos_list-np.swapaxes(separation_matrix, 0, 1)), 0, 1),N)
    separation_matrix = np.linalg.norm(separation_matrix,axis=2)
    neighbour_matrix = np.where(separation_matrix <= 1,separation_matrix,0)
    
    return_neighbour_matrix = neighbour_matrix.copy()
    return_neighbour_matrix[np.tril_indices(n=N**2, k=0)]=0
    
    spin_neighbour_matrix = np.multiply(neighbour_matrix,spin_list)
    return_spin_neighbour_matrix = spin_neighbour_matrix.copy()
    spin_neighbour_matrix[np.tril_indices(n=N**2, k=0)]=0

    energy_matrix = -1*np.multiply(spin_neighbour_matrix,spin_list.reshape(-1,1))
    energy = np.sum(energy_matrix)
    
    
    
    return energy,return_spin_neighbour_matrix,return_neighbour_matrix

def total_energy_in_step(spin_m):
    
    spin_list = spin_m.reshape(1,N**2)
    spin_neighbour_matrix = np.multiply(neighbour_matrix,spin_list)
    energy_matrix = -1*np.multiply(spin_neighbour_matrix,spin_list.reshape(-1,1))
    energy = np.sum(energy_matrix)
    
    return energy


def pair_energy_calc(i,j):
     #energy change locally
     
     e = 2*spin[i,j]*(spin[(i+1)%N,j]+
                     spin[(i-1)%N,j]+
                     spin[i,(j+1)%N]+
                     spin[i,(j-1)%N])


     return e

def change_spin(e,i,j,count=0):
     
     if e <= 0:
        spin[i,j] *= -1
                #neighbour_matrix[:,count] *= -1
     elif np.random.random() <= np.exp(-e/kT):
        spin[i,j] *= -1
                #neighbour_matrix[:,count]  *= -1

def glauber_dynamic_conventional():
    #select a random co-ordinate
    i,j = np.random.randint(0,N, size=2)
    #Get energy change locally
    #e = pair_energy_calc(*pos)
    
    
    
    e = 2*spin[i,j]*(spin[(i+1)%N,j]+
                     spin[(i-1)%N,j]+
                     spin[i,(j+1)%N]+
                     spin[i,(j-1)%N])
    #change spixn
    if e <= 0:
      spin[i,j] *= -1
       
                #neighbour_matrix[:,count] *= -1
    elif np.random.random() <= np.exp(-e/kT):
       spin[i,j] *= -1
                #neighbour_matrix[:,count]  *= -1
               
def kawasaki_dynamic_conventional():
    i,j,l,m = np.random.randint(0,N, size=4)
   
    if spin[i,j] != spin[l,m]  :
        if np.linalg.norm(np.subtract([i,j],[l,m]))%N != 1:
            e = 2*(pair_energy_calc(i,j) + pair_energy_calc(l,m))    
        else:
            e = 2*(pair_energy_calc(i,j) + pair_energy_calc(l,m)) - 4
            
        change_spin(e,[i,l],[j,m])  
def animation_func(sweep):
        plt.cla()
        plt.title(sweep)
        im = plt.imshow(spin, cmap='hot',animated=True)
        plt.draw()
        plt.pause(0.0001)
            
#calculate total energy, and get a neighbour interaction matrix
E, spin_matrix, neighbour_matrix = total_energy(spin,N)
print("The total energy of the system is {}".format(E))     
spin_matrix = sparse.csr_matrix(spin_matrix)

if animation:
    #initialise plot animation
    fig = plt.figure()
    im=plt.imshow(spin,cmap='hot', animated=True)   
        
  
mag_sus = []      
energies = []  

for kT in np.arange(kT,final_temp+0.1,step_size):
#update loop here - for Glauber dynamics
    for n in tqdm(range(nstep)):
        for _ in range(N**2):
            if dyanmics == 'g': 
                glauber_dynamic_conventional()
            elif dyanmics == 'k': 
                kawasaki_dynamic_conventional()
                        
    #occasionally plot or update measurements, eg every 10 sweeps
        if(n%10==0): 
            if animation:
                animation_func(n)
        
           
            if ((n >= 100) & (kT == 1)) or (kT > 1):
                mag_sus.append(np.sum(spin))
                energies.append(total_energy_in_step(spin))
                    
    av_E = np.mean(energies)
    err_E = np.std(energies)/np.sqrt(N**2)
    av_M = np.mean(mag_sus)
    err_M = np.std(mag_sus)/np.sqrt(N**2)
    C_v =  np.var(energies)/(N**2) 
    err_C_v = np.std(np.square(energies-av_E)/(N**2))/np.sqrt(N**2)
    


     


    # Open a file with access mode 'a'
    file_object = open('energy.dat', 'a')

    file_object.write('{} {} {} {} {} {} {}'.format(kT, av_E, err_E, av_M, err_M, C_v, err_C_v))

    file_object.close()

