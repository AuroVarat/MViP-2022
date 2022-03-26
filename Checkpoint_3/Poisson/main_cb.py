from os import system
import sys
from cv2 import RHO
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pb
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
np.random.seed(10)


N = 50
c_N = N//2
nstep = int(9e+3)+1
dx = 1.
e0 = 1.
mode = "sor"


#calc
s = slice(1,N+1)
r1by6 = 1/6
res = 11

w_list = np.linspace(1.45,2,res)

#initialization
system = np.zeros((N+2,N+2,N+2)) 

rho = np.zeros((N+2,N+2,N+2))
rho[c_N,c_N,c_N] = 1
#rho = np.pad(rho, pad_width=1)
#rho[c_N-5:c_N+5,c_N-5:c_N+5,c_N-5:c_N+5] = np.random.choice([1,0.5],(10,10,10))

#masking
select_black =  np.ones((N,N,N),dtype=bool)
select_black[1::2,1::2,1::2] = False
select_black[::2,::2,::2] = False

select_white = np.invert(select_black)

select_black = np.pad(select_black,pad_width=1)

select_white = np.pad(select_white,pad_width=1)




def update_gauss_seidel(sor=False,w=0):
    global system
    prev_system = system.copy()
    if np.isnan(system).any():
        
        raise Exception("NaN value encountered.")
    system = r1by6*(    
                        np.pad
                        ( 
                            np.roll( system,1,axis=0)[s,s,s]+ 
                            np.roll(system,-1,axis=0)[s,s,s] + 
                            np.roll(system,1,axis=1)[s,s,s] + 
                            np.roll(system,-1,axis=1)[s,s,s] +
                            np.roll(system,1,axis=2)[s,s,s]+ 
                            np.roll(system,-1,axis=2)[s,s,s],pad_width=1
                        )+ rho
                        
                    )
    
    system[select_black] = prev_system[select_black]
    
    temp = system.copy()
    
    system = r1by6*(    
                        np.pad
                        ( 
                            np.roll( system,1,axis=0)[s,s,s]+ 
                            np.roll(system,-1,axis=0)[s,s,s] + 
                            np.roll(system,1,axis=1)[s,s,s] + 
                            np.roll(system,-1,axis=1)[s,s,s] +
                            np.roll(system,1,axis=2)[s,s,s]+ 
                            np.roll(system,-1,axis=2)[s,s,s],pad_width=1
                        )+ rho
                        
                    )
    
    system[select_white] = temp[select_white]
 
    
    
    if sor:
        system *= w
        system += prev_system*(1-w) 
        
    return np.sum(np.abs(system-prev_system))

def update_jacobian():
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

def potential_calc():
    return  np.pad( 
                            np.roll(system,1,axis=0)[s,s,s]-system[s,s,s]+ 
                            np.roll(system,1,axis=1)[s,s,s]-system[s,s,s] + 
                            np.roll(system,1,axis=2)[s,s,s]-system[s,s,s] 
                            ,pad_width=1
            )
    

if mode == "gaussian":
    #gauss-seidel
    for sweep in  pb(range(nstep)):
        
        error = update_gauss_seidel(err=True)
        if np.isclose(error,0,atol=1e-3):
            break 
        
    plt.imshow(system[0:N+2,0:N+2,c_N],interpolation='gaussian')
    plt.colorbar()
    plt.show()

elif mode == 'jacobian':
    for sweep in  pb(range(nstep)):
        
        error = update_jacobian()
        if np.isclose(error,0,atol=1e-3):
            break 
        
    plt.imshow(system[0:N+2,0:N+2,c_N],interpolation='gaussian')
    plt.colorbar()
    plt.show()
    
elif mode == "sor":
    #Overrelaxation with Gauss-Seidel
    data = np.zeros((res,2))

    for index in pb(range(res)):
        
        system = np.zeros((N+2,N+2,N+2)) 
        w_i = w_list[index]
   
        for sweep in range(nstep):
            error = update_gauss_seidel(sor=True,w=w_i)
            
            if np.isclose(error,0,atol=1e-3):
                data[index] = w_i,sweep
            
                break 

    np.savetxt("Checkpoint_3/Poisson/data/output_sor.dat",data)
    plt.plot(data[:,0],data[:,1])
    plt.show()









# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# z,x,y = system.nonzero()
# # k = system.flatten()

# p = ax.scatter(x,y,z,c='red', s=2,alpha=1)
# fig.colorbar(p, ax = ax, shrink = 0.5, aspect = 5)
# plt.show()
# np.savetxt("Checkpoint_3/output_bubble_final.dat",system)