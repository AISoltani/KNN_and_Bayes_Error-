# Import Libraries

import numpy as np
from data_gen import data_gen
from Bayes import Bayes
from scipy.spatial.distance import cdist
from knn import knn

n = 6000
d = 2
b = 0
w = np.zeros([1,d-1])
w = np.insert(w,0,[1])
w = np.reshape(w,(1,d))
sigma_list = [0.01,0.1,1,10]

for sigma in sigma_list:
    
    trainX,trainy =  data_gen (n , d, sigma, w, b)
    testX,testy =  data_gen (n , d, sigma, w, b)
    
    # Bayes
    Bayes_error = Bayes (trainX,trainy,w,b)
    
    # knn
    k = 1
    dist = 2 # dist norm => if dist = 2, it means 2 norm
    y_hat,error = knn(trainX, trainy, testX, k, dist, testy)
    
    print("sigma = ", sigma)
    print ('Bayes_eEror: ',Bayes_error)
    print("K ",k,"Error : ", error)
