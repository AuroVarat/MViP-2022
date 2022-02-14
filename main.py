#%%
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from numba import jit
#%%
#terminal input
N = int(input("System Size: ")) #System Size
kT = int(input("Temperature: ")) #Temperature
dyanmics = str(input("Dynamics Algorithm(g or k): ")) #Alogorithm
animation = bool(int(input("Animation: "))) #Show Animation?
analysis_mode = bool(int(input("Analysis Mode: "))) #Do analysis?
#Choose Analysis Parameters
if analysis_mode:
    final_temp =  int(input("Final Temperature: "))
    step_size = float(input("Step Size: "))
else:
    final_temp = kT
    step_size = 1
#%%    
#Global Variable
J=1.0
nstep=10000 #Sweeps
pnum = N**2 #Number of cells
np.random.seed(10) 
spin = np.random.choice([-1,1],[N,N]) #initialise spins randomly
index_matrix = np.random.randint(0,N, size=(pnum*10000,2,2)) #initialise random values
#Max Probability for Positive Energy changes in the system
e4 = np.exp(-4/kT)
e8 = np.exp(-8/kT)
e12 = e4**3
e16 = e4**4



#%%
def mic(vector,box_size):
    """
        Finds the image of the points with MIC

        :param vector: Any kind of vectorized element in list or on its own
        :param box_size: size of the simulation box

        :return MIC enforced vector image: vector
        
    """
    return np.subtract(np.mod(np.add(vector,np.multiply(box_size,0.5)),box_size),np.multiply(box_size,0.5))

def total_energy(spin,N):
    """
        Finds the initial energy of the system

        :param spin: Array containing the spins 
        :param N: number of cells 

        :return energy: float
        :return return_neighbour_matrix - 
            Array shows the neighbour cells of each cell every row: Numpy Array
        
    """
    spin_list = spin.reshape(1,pnum)

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
@jit(nopython=True)
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

def glauber_dynamic_conventional(i):
    #select a random co-ordinate
    i,j = index_matrix[i]
    #Calculate Energy change  
    
    e = 2*spin[i,j]*(spin[(i+1)%N,j]+spin[(i-1)%N,j]+spin[i,(j+1)%N]+spin[i,(j-1)%N])
    #Change Spin
    if e <= 0:
        spin[i,j] *= -1       
    elif ((e == 4) & (np.random.random() <= e4)) or ((e == 8) & (np.random.random() <= e8)):
        spin[i,j] *= -1
    

               
def kawasaki_dynamic_conventional(s1,s2,i,j,l,m):
    
   
    
        e = 2*(s1*(spin[(i+1)%N,j]+spin[(i-1)%N,j]+spin[i,(j+1)%N]+spin[i,(j-1)%N])+
                   s2*(spin[(l+1)%N,m]+spin[(l-1)%N,m]+spin[l,(m+1)%N]+spin[l,(m-1)%N]))
        if np.linalg.norm(np.subtract([i,j],[l,m]))%N == 1:
            e -= 4
        
        if e <= 0:
            spin[[i,l],[j,m]] *= -1   
        elif ((e == 8) & (np.random.random() <= e8)) or ((e == 16) & (np.random.random() <= e16))  or ((e == 12) & (np.random.random() <= e12)):
            spin[[i,l],[j,m]] *= -1 
            
def animation_func(sweep):
        plt.cla()
        plt.title(sweep)
        im = plt.imshow(spin, cmap='hot',animated=True)
        plt.draw()
        plt.pause(0.0001)
#%%            
#calculate total energy, and get a neighbour interaction matrix
E, spin_matrix, neighbour_matrix = total_energy(spin,N)
print("The total energy of the system is {}".format(E))     

if animation:
    #initialise plot animation
    fig = plt.figure()
    im=plt.imshow(spin,cmap='hot', animated=True)   
        
  
data = np.zeros((1000,2))
#%%
for kT in np.arange(kT,final_temp+0.1,step_size):
#update loop here - for Glauber dynamics
    for n in tqdm(range(nstep)):
        for p in range(N**2):
            [i,j],[l,m] = index_matrix[p+n*pnum]
            s1 = spin[i,j]
            s2 = spin[l,m]
            if s1 != s2  :
                kawasaki_dynamic_conventional(s1,s2,i,j,l,m)
            # if dyanmics == 'g': 
            #     glauber_dynamic_conventional(p+n*(N**2))
            # elif dyanmics == 'k': 
            #     kawasaki_dynamic_conventional()
                        
    #occasionally plot or update measurements, eg every 10 sweeps
        if(n%10==0): 
            if animation:
                animation_func(n)
        
            if ((n >= 100) & (kT == 1)) or (kT > 1):
                
                data[n//10]=[total_energy_in_step(spin),np.sum(spin)]
              
    mean_data = np.mean(data,axis=0)    
    err_data  = np.std(data,axis = 0)/1000     
    av_E = mean_data[0]
    enegies = data[:,0]
    C_v =  np.var(enegies)/(pnum*(kT**2)) 
    err_C_v = np.std(np.square(enegies-av_E)/(pnum*(kT**2)))/1000
    


     


    # Open a file with access mode 'a'
    file_object = open('energy.dat', 'a')

    file_object.write('\n{} {} {} {} {} {} {}'.format(kT, av_E, err_data[0],mean_data[1], err_data[1], C_v, err_C_v))

    file_object.close()







# %%

# %%

# %%

# %%
