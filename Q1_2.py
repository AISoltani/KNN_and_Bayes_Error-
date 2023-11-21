# Import Libraries

import numpy as np
from data_gen import data_gen
from Bayes import Bayes
import matplotlib.pyplot as plt

n = 6000
d = 2
sigma = 1
b = 0
w = np.zeros([1,d-1])
w = np.insert(w,0,[1])
w = np.reshape(w,(1,d))
N = [10,50,100,200,300,500,800,1000,2000,5000,8000,10000,20000,50000,100000,200000,500000,1000000]
Bayes_Error = []
Bayes_Error_A = []
for n in N:
    X,y =  data_gen (n , d, sigma, w, b)
    Bayes_error = Bayes (X,y,w,b)
    Bayes_Error.append(Bayes_error)
    print ('% Numerical Bayes_error: ',Bayes_error)

    Bayes_error_A = np.mean(1/(1 + np.exp(np.abs(X[:,0])))) * 100 
    Bayes_Error_A.append(Bayes_error_A)
    print ('% Analitical Bayes_error: ',Bayes_error_A)

Bayes_Error_A  = np.array(Bayes_Error_A)
Bayes_Error = np.array(Bayes_Error) 
N = np.array(N)
                          
plt.figure()
plt.plot(N,Bayes_Error, linewidth = 4, label = 'Numerical Bayes_error')
plt.plot(N,Bayes_Error_A, label = 'Analytical Bayes_error')
plt.xscale('log')
plt.legend()
plt.xlabel('Number of data')
plt.ylabel('Error %')
plt.show()
############ data-gen functin
# def  data_gen (n , d, sigma, w, b):
#     X = np.random.uniform(-1, 1, (n, d))
#     U = np.random.uniform(-1, 1, (1, n))
#     B = b * np.ones([n,1])
    
#     SGM = 1/(1 + np.exp(-(np.dot(w,X.T) + B.T)/sigma))
#     mask = SGM <= U
#     y = np.zeros_like(mask,dtype=int)
#     y[mask] = 1
#     y[~mask] = -1
#     return X,y

############ Bayes functin
# import numpy as np
# from data_gen import data_gen
# def Bayes (X,y,w,b):
#     B = b * np.ones([n,1])
#     mask_Bayes = (np.dot(w,X.T) + B.T) >= 0
#     y_hat = np.zeros_like(mask_Bayes,dtype=int)
#     y_hat[mask_Bayes] = 1
#     y_hat[~mask_Bayes] = -1
    
#     np.array(np.where(y != y_hat))
#     error = np.array(np.where(y != y_hat)).shape[1]/n * 100
#     return     print(error)
