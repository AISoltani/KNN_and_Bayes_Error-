import numpy as np
from data_gen import data_gen


def Bayes (X,y,w,b):
    n = X.shape[0]
    B = b * np.ones([n,1])
    mask_Bayes = (np.dot(w,X.T) + B.T) >= 0
    y_hat = np.zeros_like(mask_Bayes,dtype=int)
    y_hat[mask_Bayes] = 1
    y_hat[~mask_Bayes] = -1
    
    np.array(np.where(y != y_hat))
    error = np.array(np.where(y != y_hat)).shape[1]/n * 100
    return    error
