import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

A = 50
pub = np.random.choice([0,1,-1],[A,A])
fig = plt.figure()
im=plt.imshow(pub,cmap='hot', animated=True)  
plt.show()