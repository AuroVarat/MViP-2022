import numpy as np
import numpy as np

a = np.arange(5)

b =np.random.choice(a,(2,5))
print(b)
print(np.sum(b,axis=1))
