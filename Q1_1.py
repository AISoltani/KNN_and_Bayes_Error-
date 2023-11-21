# Import Libraries

import numpy as np
from data_gen import data_gen

n = 6000
d = 2
sigma = 1
b = 0
w = np.zeros([1,d-1])
w = np.insert(w,0,[1])
w = np.reshape(w,(1,d))

X_1,y_1 =  data_gen (n , d, sigma, w, b)
print('X = ',X_1)
print('y = ', y_1)

# def  data_gen (n , d, sigma, w, b):
#     X = np.random.uniform(-1, 1, (n, d))
#     U = np.random.uniform(0, 1, (1, n))
#     B = b * np.ones([n,1])
    
#     SGM = 1/(1 + np.exp(-(np.dot(w,X.T) + B.T)/sigma))
#     mask = U <= SGM
#     y = np.zeros_like(mask,dtype=int)
#     y[mask] = 1
#     y[~mask] = -1
#     return X,y
