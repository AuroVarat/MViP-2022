from cmath import nan
from dataclasses import replace
from operator import index
from cv2 import bilateralFilter
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys



def main(A=50, n=1100, p1=0.8, p2=0.1, p3=0.01, fim=0., animation = 0):
    """
    -1: recovered
    +0: susceptible
    +1: infected
    """
    
    N = A**2
    x = np.arange(0,A)
    x2d = np.array(np.meshgrid(x, x)).T.reshape(-1,2)
    
    rng = np.random.default_rng()
    immune = rng.choice(x2d,replace=False,size=int(fim*N))
    
    population = np.random.choice([1,0,-1],[A,A])
    index_matrix = np.random.randint(0,A, size=(N*n,2)) #initialise random value
    population[immune[:,0],immune[:,1]] = 2
    
    
    
    
    # inds = index_matrix==immune[:,None]
    # row_sums = inds.sum(axis = 2)
    # _, j = np.where(row_sums == 2)
    
    # index_matrix = np.delete(index_matrix,j,axis=0)

    if animation:
        cMap = ListedColormap(['lightgreen', 'yellow','red','lightblue'])
        plt.figure()
        im=plt.imshow(population,cmap=cMap,animated=True)
        plt.title("SIRS Model") 
        
        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['Recovered','Susceptible','Infected','Immunized']):
            cbar.ax.text(2.5, (3 * j -2) / 4.0, lab, ha='left', va='center')
        cbar.ax.get_yaxis().labelpad = 15

    def update(rn):
        #select a random co-ordinate
        ind = index_matrix[rn]
        i,j = ind
        state = population[tuple(ind)]
        
        if  state == -1 and np.random.random() < p3 :
            #recovered to susceptible
                state += 1
        elif state == 0 and np.random.random() < p1 and (population[(i+1)%A,j] ==1 or population[(i-1)%A,j] ==1 or population[i,(j+1)%A] ==1 or population[i,(j-1)%A] ==1):
            #susceptible to infected
                state += 1
        elif state == 1 and np.random.random() < p2 :
            #infected to recovered
            state -= 2
            
        population[tuple(ind)] = state
    def bootstrap(data_list,resample_n,scale):
        res = np.random.choice(data_list,(resample_n,10000))
        fres = np.var(res,axis=1)/scale

        return np.var(fres)**(1/2)
     
     
        
    if animation:
        for frame in range(n):
            for loop in range(N):
            
                update(loop + frame*N)

            plt.cla()
            plt.title("SIRS Model with immunized fraction: {}".format(fim))
            plt.imshow(population,cmap=cMap,animated=True,vmin=-1,vmax=2)
            plt.draw()
            plt.pause(0.0001)
    else:
        infc_array = np.empty(n-100)
        
        for frame in range(n):
            for loop in range(N):
                update(loop + frame*N)
                
                
            if frame >= 100:
                inf = np.where(population == 1)[0].size
                infc_array[frame-100] = inf
                if inf == 0:
                    return 0
                    
        else:
            return np.nanmean(infc_array)/N
              
      
            
                

        
    

if __name__ == "__main__":
    if(len(sys.argv) == 7):
        A =int(sys.argv[1]) 
        p1 = float(sys.argv[2])
        p2 = float(sys.argv[3])
        p3 = float(sys.argv[4])
        fim = float(sys.argv[5])
        animation = bool(int(sys.argv[6])) 
        
    elif (len(sys.argv) == 5):
        A =int(sys.argv[1]) 
        stype = str(sys.argv[2])
        if stype == "abs":
            #adsolving state
            p1 = 0.5
            p2 = 0.6
            p3 = 0.1
            
        elif stype == "wvs":
            #wavey
            p1 = 0.8
            p2 = 0.1
            p3 = 0.01
            
        elif stype == "dyeq":
            #dya
            p1 = 0.5
            p2 = 0.5
            p3 = 0.5
        fim = float(sys.argv[3])
        animation = bool(int(sys.argv[4]))
    else:
        print ("Usage python file.py 'A' p1 p2 p3 fim animation")
        print ("or")
        print ("Usage python file.py 'A' SimulationType fim animation")
        sys.exit()


   
    
    main(A=A,n=1000,p1=p1,p2=p2,p3=p3,fim = fim,animation=animation)
 

