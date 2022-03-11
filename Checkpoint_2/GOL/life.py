from matplotlib import pyplot
from main import main
import numpy as np
from tqdm import tqdm

# n = 6000

# sites = np.empty(n)
# for i in tqdm(range(n)):
#     sites[i] = main()

sites = np.genfromtxt("Checkpoint_2/GOL/hist.dat")
#np.savetxt("Checkpoint_2/GOL/hist-big.dat", sites)
pyplot.hist(sites,bins=70,range=(0,np.nanmax(sites))) 
pyplot.show() 
#np.savetxt("Checkpoint_2/GOL/hist-big.dat", sites)