import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pb
import sys


r1by6 = 1/6
dx = 1.
e0 = 1.
u = 1.
tol = 1e-3
w_optimized = 1.89

def create_checkerboard(shape):
        mask =  np.bool_(np.indices(shape).sum(axis=0) % 2)
        return mask, ~mask


def main(N=50,nstep = int(250),mode = "gaussian",field = "magnetic"):
    global potential
  
    c_N = N//2
    Nb = N+2 
    s = slice(1,N+1)
    dims = np.full(3,Nb)
    
    potential, reset, system = np.zeros((3,*dims)) 
    select_black, select_white = create_checkerboard(dims)
    
    if field =="electric":
        system[c_N,c_N,c_N] = 1
        npad = ((1,1), (1,1),(1, 1))
        active_region = (s,s,s)
    elif field == "magnetic":
        system[c_N,c_N] = 1
        npad = ((1,1), (1,1),(0, 0))
        active_region = (s,s,)
        

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
            potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
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
            potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
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
            #overelac
            potential *= w
            potential += prev_potential*(1-w) 
            #mask
            potential[select_black] = prev_potential[select_black]
            #enforce boundary
            potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
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
            #overrelax
            potential *= w
            potential += prev_potential*(1-w)
            #mask
            potential[select_white] = temp[select_white]
            #enforce boundary
            potential = np.pad(potential[active_region], npad, mode='constant', constant_values=0)
           
            
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
   
        Fx = (np.roll(potential,1,axis=0)-np.roll(potential,-1,axis=0))[s,s,c_N].ravel()/-2*dx
        Fy = (np.roll(potential,1,axis=1)-np.roll(potential,-1,axis=1))[s,s,c_N].ravel()/-2*dx
        Fz = (np.roll(potential,1,axis=2)-np.roll(potential,-1,axis=2))[s,s,c_N].ravel()/-2*dx
        return Fx,Fy,Fz

    def savedata():
        pos_list = np.array(np.meshgrid(np.arange(1,N+1), np.arange(1,N+1))).T.reshape(-1,2)
        dis_list = np.linalg.norm(pos_list - c_N,axis=1)
        
        pot_list = potential[s,c_N,s].ravel()
        
        if field == "electric":
            Ex,Ey,Ez = field_calc()
            np.savetxt("Checkpoint_3/Poisson/data/{}_{}_output.dat".format(N,field),np.c_[pos_list,dis_list,pot_list,Ex,Ey,Ez])
        elif field == "magnetic":
            My,Mx,_ = field_calc()
            np.savetxt("Checkpoint_3/Poisson/data/{}_{}_output.dat".format(N,field),np.c_[pos_list,dis_list,pot_list,Mx,-My])
        np.savetxt("Checkpoint_3/Poisson/data/{}_2D_{}_output.dat".format(N,field),potential[0:Nb,0:Nb,c_N])
        
    def plot_potentialSlice():
        plt.imshow(potential[0:Nb,c_N,0:Nb],interpolation='gaussian')
        plt.colorbar()
        plt.show()
        
    if mode == "gaussian":
        #gauss-seidel
        for sweep in  pb(range(nstep)):
            if np.isclose(update_sor(w=w_optimized),0,atol=tol):
                break 
        
        savedata()
        plot_potentialSlice()    
        
    elif mode == 'jacobian':
    
        for sweep in  pb(range(nstep)):
            if np.isclose(update_jacobian(),0,atol=1e-3):
                break 

        plot_potentialSlice()

    elif mode == "sor":
        res = 20
        w_list = np.linspace(1,2,res,endpoint=False)
        data = np.zeros((res,2))
    
        for index in pb(range(res)):
            
            potential = reset.copy()
            w_i = w_list[index]
            
            for sweep in range(nstep):
                if np.isclose(update_sor(w=w_i),0,atol=1e-3):
                    data[index] = w_i,sweep
                    break 
        # np.savetxt("Checkpoint_3/Poisson/data/output_sor.dat",data)
        # plt.scatter(data[:,0],data[:,1])
        # plt.show()
        
        

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
    