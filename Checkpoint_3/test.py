import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm as pb


N = 100
phi0 = 0


kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                    [0, 1, 0]])
   

a = np.random.uniform(-0.1+phi0,0.1+phi0,(N,N))  

for i in pb(range(int(1e+6))):
    a =  np.convolve(a, kernel, mode='same')
    # a = (np.roll(a,1,axis=0) + 
    #     np.roll(a,-1,axis=0) + 
    #     np.roll(a,1,axis=1) + 
    #     np.roll(a,-1,axis=1))
     