import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pb
import sys


r1by6 = 1/6
res = 20
dx = 1.
e0 = 1.
u = 1.
npad = ((1,1), (1,1),(0, 0))

def create_checkerboard(shape):
    mask =  np.bool_(np.indices(shape).sum(axis=0) % 2)
    return mask, ~mask

def update_gauss_seidel():
        global potential
        
        prev_potential = potential.copy()
        
        potential = r1by6*(    
                                np.roll( potential,1,axis=0)+ 
                                np.roll(potential,-1,axis=0) + 
                                np.roll(potential,1,axis=1)+ 
                                np.roll(potential,-1,axis=1)+
                                np.roll(potential,1,axis=2)+ 
                                np.roll(potential,-1,axis=2)
                            + system
                            
                        )

        potential[select_black] = prev_potential[select_black]
        potential = np.pad(potential[s,s,s], npad, mode='constant', constant_values=0)
        #potential = np.pad(potential[s,s,s],pad_width=1)
        temp = potential.copy()
        
        
        
        potential = r1by6*(    
                            np.roll( potential,1,axis=0)+ 
                            np.roll(potential,-1,axis=0) + 
                            np.roll(potential,1,axis=1)+ 
                            np.roll(potential,-1,axis=1)+
                            np.roll(potential,1,axis=2)+ 
                            np.roll(potential,-1,axis=2)
                        + system
                        
                    )
        
        potential[select_white] = temp[select_white]
        potential = np.pad(potential[s,s,s], npad, mode='constant', constant_values=0)
        #potential = np.pad(potential[s,s,s],pad_width=1)
        
        return np.sum(np.abs(potential-prev_potential))

def update_sor(w=0):
        global potential
        
        prev_potential = potential.copy()
        
        potential = r1by6*(    
                                np.roll( potential,1,axis=0)+ 
                                np.roll(potential,-1,axis=0) + 
                                np.roll(potential,1,axis=1)+ 
                                np.roll(potential,-1,axis=1)+
                                np.roll(potential,1,axis=2)+ 
                                np.roll(potential,-1,axis=2)
                            + system
                            
                        )
        
        
        

        potential *= w
        potential += prev_potential*(1-w) 

        potential[select_black] = prev_potential[select_black]
        potential = np.pad(potential[s,s,:], npad, mode='constant', constant_values=0)
        #potential = np.pad(potential[s,s,s],pad_width=1)
        
        temp = potential.copy()
    
        
        
        potential = r1by6*(    
                                np.roll( potential,1,axis=0)+ 
                                np.roll(potential,-1,axis=0) + 
                                np.roll(potential,1,axis=1)+ 
                                np.roll(potential,-1,axis=1)+
                                np.roll(potential,1,axis=2)+ 
                                np.roll(potential,-1,axis=2)
                            + system
                            
                        )
        
        
    
        potential *= w
        
        potential += prev_potential*(1-w)
        
        potential[select_white] = temp[select_white]
        potential = np.pad(potential[s,s,:], npad, mode='constant', constant_values=0)
        #potential = np.pad(potential[s,s,s],pad_width=1)
        
        return np.sum(np.abs(potential-prev_potential))

def update_jacobian():  
        global potential  
        
        prev_potential = potential.copy()

        potential = r1by6*(    
                            np.roll( potential,1,axis=0)+ 
                            np.roll(potential,-1,axis=0) + 
                            np.roll(potential,1,axis=1)+ 
                            np.roll(potential,-1,axis=1)+
                            np.roll(potential,1,axis=2)+ 
                            np.roll(potential,-1,axis=2)
                        + system
                        
                    )
        potential = np.pad(potential[s,s,s],pad_width=1)

        
        return np.sum(np.absolute(potential-prev_potential))


    
def field_calc():
    Fx = (np.roll(potential,1,axis=1)-np.roll(potential,-1,axis=1))[s,s,c_N].ravel()/2*dx
    Fy = (np.roll(potential,1,axis=0)-np.roll(potential,-1,axis=0))[s,s,c_N].ravel()/2*dx
    Fz = (np.roll(potential,1,axis=2)-np.roll(potential,-1,axis=2))[s,s,c_N].ravel()/2*dx
    return Fx,Fy,Fz


def main(N=50,nstep = int(9e+3)+1,mode = "gaussian",field = "magnetic"):
    global potential, system, c_N, Nb, s, dims, select_black, select_white
  
    c_N = N//2
    Nb = N+2 
    s = slice(1,N+1)
    dims = np.full(3,Nb)
    
    potential = np.zeros(dims) 
    reset = np.zeros(dims) 

    system = np.zeros(dims)
    
    if field =="electric":
        system[c_N,c_N,c_N] = 1
    elif field == "magnetic":
        system[c_N,c_N] = 1
        
    select_black, select_white = create_checkerboard(dims)


    if mode == "gaussian":
        #gauss-seidel
        for sweep in  pb(range(nstep)):
            
            error = update_sor(w=1.89)
            
            if np.isclose(error,0,atol=1e-3):
                break 
            
        pos_list = np.array(np.meshgrid(np.arange(1,N+1), np.arange(1,N+1))).T.reshape(-1,2)
        dis_list = np.linalg.norm(pos_list - c_N,axis=1)
        
        sys_list = potential[s,c_N,s].ravel()
      
        Ex,Ey,_ = field_calc()

        np.savetxt("Checkpoint_3/Poisson/data/{}_2D_{}_output.dat".format(N,field),potential[0:Nb,0:Nb,c_N])
        np.savetxt("Checkpoint_3/Poisson/data/{}_{}_output.dat".format(N,field),np.c_[pos_list,dis_list,sys_list,Ex,-Ey])
        
        plt.imshow(potential[0:Nb,c_N,0:Nb],cmap="Pastel1",interpolation='gaussian')
        plt.colorbar()
        plt.show()
        
        # plt.scatter(np.log(dis_list),np.log(sys_list))
        # plt.show()
        
    elif mode == 'jacobian':
     
        for sweep in  pb(range(nstep)):
            if np.isclose(update_jacobian(),0,atol=1e-3):
                break 

        plt.imshow(potential[0:Nb,c_N,0:Nb],interpolation='gaussian')
        plt.colorbar()
        plt.show()

    elif mode == "sor":
        w_list = np.linspace(1,2,res,endpoint=False)
        data = np.zeros((res,2))
    
        for index in pb(range(res)):
            
            potential = reset
            w_i = w_list[index]
            
            for sweep in range(nstep):
                if np.isclose(update_sor(w=w_i),0,atol=1e-3):
                    data[index] = w_i,sweep
                    break 
        
                
            

        # np.savetxt("Checkpoint_3/Poisson/data/output_sor.dat",data)
        plt.scatter(data[:,0],data[:,1])
        plt.show()
        
        

   
    
if __name__ == "__main__":

    if(len(sys.argv) == 5):
        N =int(sys.argv[1]) 
        nstep = int(sys.argv[2])+1
        mode = sys.argv[3] # gaussian / jacobian / sor
        field= sys.argv[4] # magnetic / electric
        
        main(N=N,nstep = nstep,mode = mode,field = field)
    elif(len(sys.argv) == 1) :
        print("No input provided. Default values will be used") 
        main()
    else:
        print("Error. Required Usage: N nstep mode field")
    