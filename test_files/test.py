import numpy as np
from scipy import sparse
# spin = np.random.randint(0,2, size=(5, 5),dtype=int)
# spin[spin == 0] = -1
# print(np.indices(spin))
from tqdm import tqdm
from numba import jit

def mic(vector,box_size):
    """
        Finds the image of the points with MIC

        :param vector: Any kind of vectorized element in list or on its own
        :param box_size: size of the simulation box

        :return MIC enforced vector image: vector
        
    """
    
    return np.subtract(np.mod(np.add(vector,np.multiply(box_size,0.5)),box_size),np.multiply(box_size,0.5))
np.random.seed(10)
N = 50

spin = np.random.choice([-1,1],[N,N])

pos_list = np.array(np.meshgrid(np.arange(0,N), np.arange(0,N))).T.reshape(-1,2)
separation_matrix = np.ones((N**2,N**2,2))*pos_list
separation_matrix = mic(np.swapaxes((pos_list-np.swapaxes(separation_matrix, 0, 1)), 0, 1),N)
separation_matrix = np.linalg.norm(separation_matrix,axis=2)
nei = np.where(separation_matrix <= 1,separation_matrix,0)
separation_matrix = nei.copy()
separation_matrix[np.tril_indices(n=N**2, k=0)]=0




spin_list = spin.reshape(1,N**2)
spin_matrix = np.ones((N**2,N**2))*spin_list
spin_neighbor_matrix = np.multiply(nei,spin_list)
energy_matrix = -1*spin_neighbor_matrix*spin_matrix
energy = np.sum(energy_matrix)


print("Test")
print(spin_neighbor_matrix.T)




@jit(nopython=True)
#spin_neighbor_matrix = sparse.csr_matrix(spin_neighbor_matrix).todense()
def doLoop():
    
        for i in range(N**2):
            np.negative(spin_neighbor_matrix[:,i])
            np.sum(spin_neighbor_matrix[i])
        #spin_neighbor_matrix[i]
        # z = np.ones(N**2)
        # z[i] = -1
        #np.multiply(spin_neighbor_matrix,z)
for i in tqdm(range(10000)):        
    doLoop()



