import numpy as np

def  data_gen (n , d, sigma, w, b):
    
    X = np.random.uniform(-1, 1, (n, d))
    U = np.random.uniform(0, 1, (1, n))
    B = b * np.ones([n,1])
    SGM = 1/(1 + np.exp(-(np.dot(w,X.T) + B.T)/sigma))
    mask =  U <= SGM 
    y = np.zeros_like(mask,dtype=int)
    y[mask] = 1
    y[~mask] = -1
    return X,y

