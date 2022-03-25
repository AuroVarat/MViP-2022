import enum
import numpy as np

N = 6
a = np.full((N, N),6)
b = a.copy()

mask_one =  np.ones((N, N),dtype=bool)
mask_one[1::2,1::2] = False
mask_one[::2,::2] = False
a[mask_one]*=1
b[np.invert(mask_one)] *= 0 

print(a)
print(b)
s = slice(2,4)
print(a[s,s])
# data = np.empty((21,2))
# l = np.linspace(1,2,21)
# print(l)
# for i,j in enumerate(l):
#     data[i] = i,j
# print(data)