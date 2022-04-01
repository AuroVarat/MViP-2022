
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
# w = np.linspace(1,2,21)
# data = np.empty((21,2))
# for i,j in enumerate(w):
#     data[i] = i,j
# print(data)
# N = 6
# a = np.full((N, N),6)
# b = a.copy()

# mask_one =  np.ones((N, N),dtype=bool)
# mask_one[1::2,1::2] = False
# mask_one[::2,::2] = False
# a[mask_one]*=1
# b[np.invert(mask_one)] *= 0 

# print(a)
# print(b)
# s = slice(2,4)
# print(a[s,s])
# # data = np.empty((21,2))
# # l = np.linspace(1,2,21)
# # print(l)
# # for i,j in enumerate(l):
# #     data[i] = i,j
# # print(data)

# N  = 4
# c_N = N//2
# rho = np.zeros((N+2,N+2))
# rho[c_N,c_N] = 1
# pos_list = np.array(np.meshgrid(np.arange(0,N), np.arange(0,N))).T.reshape(-1,2)
# dis_list = np.linalg.norm(pos_list - c_N,axis=1)
# print(rho)
# print(rho[0:N,0:N])
# res = 20
# w_list = np.linspace(1,2,res,endpoint=False)
# print(print(w_list))

# def checkerboard(shape):
    
#     return np.bool_(np.indices(shape).sum(axis=0) % 2)



# print(checkerboard((2,2,2)))

a = np.ones((5,5))

# npad is a tuple of (n_before, n_after) for each dimension
npad = ((0,0), (0,0),(0, 0))
b = np.pad(a, npad, mode='constant', constant_values=0)
print(b)
# potential = np.pad(potential[s,s,s], npad, mode='constant', constant_values=0)
# #potential = np.pad(potential[s,s,s],pad_width=1)