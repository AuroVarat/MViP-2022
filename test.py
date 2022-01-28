import numpy as np

# spin = np.random.randint(0,2, size=(5, 5),dtype=int)
# spin[spin == 0] = -1
# print(np.indices(spin))

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
separation_matrix[np.tril_indices(n=N**2, k=0)]=0
separation_matrix = np.linalg.norm(separation_matrix,axis=2)
separation_matrix = np.where(separation_matrix <= 1,separation_matrix,0)

print(separation_matrix)

spin_list = spin.reshape(1,N**2)
spin_matrix = np.ones((N**2,N**2))*spin_list
spin_neighbor_matrix = np.multiply(separation_matrix,spin_list)
energy_matrix = -1*spin_neighbor_matrix*spin_matrix
energy = np.sum(energy_matrix)
print(energy)

