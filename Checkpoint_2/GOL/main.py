import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys




def main(N = 50,sweeps = 10000, dish_type = "random", mode = 0):    
    c = N//2
    
    def addGlider():
        """adds a glider with top left cell at (i, j)"""
        
        glider = np.array([[0,    0, 1],
                        [1,  0, 1],
                        [0,  1, 1]])
        dish[c-1:c+2, c-1:c+2] = glider   
    def addBlinker():
        """adds a glider with top left cell at (i, j)"""
        
        blinker = np.array([[0,  1, 0],
                        [0,  1, 0],
                        [0,  1, 0]])
        dish[c-1:c+2, c-1:c+2] = blinker
    def addComb():
        comb = np.array([[0,  1, 0],
                        [1,  0, 1],
                        [1,  0, 1],
                        [0,  1, 0]])
        dish[c-2:c+2, c-1:c+2] = comb  
    def addRP():
        """adds a R-pentonimo with top left cell at (i, j)"""

        blinker = np.array([[0,  1, 1],
                            [1,  1, 0],
                            [0,  1, 0]])
        dish[c-1:c+2, c-1:c+2] = blinker   
    
    if dish_type == "random":
            np.random.seed(None)
            dish = np.random.choice([0,1],[N,N],p=[0.8,0.2]) 
            #dish = np.genfromtxt("Checkpoint_2/GOL/noneq.dat") 
    elif dish_type == "glider":
        dish = np.zeros((N,N)) 
        addGlider()
    elif dish_type == "blinker":
        dish = np.zeros((N,N)) 
        addBlinker()
    
 
 
    def update():
        newDish = (np.roll(dish,1,axis=0) + np.roll(dish,-1,axis=0) + 
                np.roll(dish,1,axis=1) + np.roll(dish,-1,axis=1) +
                np.roll(dish,(1,1),axis=(1,0)) +  np.roll(dish,(1,-1),axis=(1,0)) +
                np.roll(dish,(-1,-1),axis=(1,0)) +  np.roll(dish,(-1,1),axis=(1,0)) 
                )
        newDish = np.where(newDish == 3, 1,0) + np.where((dish == 1) & (newDish == 2),1,0)
    
        return newDish
    
    
    def tracker(boundary,crossedBoundary):
        masses = np.nonzero(dish)

        if (49 in masses[0] or 49 in masses[1]):
     
            if crossedBoundary == False:
                boundary +=1
                
            crossedBoundary = True
            return np.zeros(2),boundary,crossedBoundary
        else:
           
            com = np.sum(masses, axis = 1)/len(masses[0]) + (boundary * 50) #center of mass position
            crossedBoundary = False
            
            return com,boundary,crossedBoundary

    
    #initialise plot animation
   
    
    if mode:
        #initialise plot animation
        plt.figure()
        plt.imshow(dish,cmap='hot', animated=True) 
        data = np.zeros((sweeps,3))
        crossedBoundary = False 
        boundary = 0
        for frame in tqdm(range(sweeps)):
            
            dish = update()
            com, boundary, crossedBoundary = tracker(boundary,crossedBoundary)
            data[frame] = [*com,np.count_nonzero(dish)]
        
            plt.cla()   
            plt.title(np.count_nonzero(dish))
            plt.imshow(dish,animated=True)
            plt.draw()
            plt.pause(0.0001)
        np.savetxt("Checkpoint_2/GOL/output-one.dat", data, delimiter=" ", header="x,y,na")
    else:

        flat_counter = 0
        sites = [0]
        for frame in range(sweeps):
            
            if flat_counter == 10:
                return frame - 10
                
                
                
            
            dish = update()
            active = np.count_nonzero(dish)
            if np.isclose(active,sites[-1],atol=2):
                flat_counter += 1
            else:
                flat_counter = 0 
            sites.append(active)
            
        else:
            
            print("No Equlibration found ")
         
            # plt.figure()
            # plt.imshow(dish,cmap='hot', animated=True) 
           
            # for frame in range(200):
            #     dish = update()
            #     plt.cla()   
            #     plt.title(np.count_nonzero(dish))
            #     plt.imshow(dish,animated=True)
            #     plt.draw()
            #     plt.pause(0.0001)
                

if __name__ == "__main__":
    if(len(sys.argv) != 5):
        print ("Usage python file.py N T dish mode")
        sys.exit()

    N=int(sys.argv[1]) 
    sweeps = int(sys.argv[2]) 
    dish_type= sys.argv[3]
    mode = bool(int(sys.argv[4])) 
    
    main(N,sweeps,dish_type,mode)