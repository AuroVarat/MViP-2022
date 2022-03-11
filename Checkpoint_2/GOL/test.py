from dataclasses import replace
from operator import index
from matplotlib.pyplot import axis
import numpy as np
import itertools
# x = np.arange(0,4)
# x2d = np.array(np.meshgrid(x, x)).T.reshape(-1,2)

population = np.random.choice([1,0,-1],[4,4])
# print(population)
# rng = np.random.default_rng()
# immune = rng.choice(x2d,replace=False,size=3)
# print(immune)
# population[immune[:,0],immune[:,1]] = 1
# print(population)
# # index_matrix =np.delete(index_matrix,np.where(index_matrix == immune),axis=0)
# # print(len(index_matrix))
print(population)
print(np.sum(population,axis=1))