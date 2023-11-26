# Import Libraries

import numpy as np
from data_gen import data_gen
from scipy.spatial.distance import cdist

def knn(trainX, trainy, testX, k, dist, testy):
    trainy = np.reshape (trainy,-1)
    testy = testy.T
    n = trainX.shape[0]
# distance of text data from the trained data
# Each row is the distance of each test data from all of the trained data
    Dist_Matrix = cdist(testX,trainX, metric='minkowski', p= dist)
    CLassy_Mins_Dis = np.zeros([k,n])
    for i in range(k):
    # Each element of following vector (MIN) is 
    # the min of each test data from trained data is determined
        MIN = np.min(Dist_Matrix,axis=1)
        MIN = np.reshape(MIN,(n,1)) # change the row vector to column vector
    # the index of those mins 
        index = np.where(Dist_Matrix == MIN) 
        index_x,index_y = index
    # change the mins to inf because if k>1 we need other mins 
    # forexample ( 3rd mins 5th mins and etc.)
        Dist_Matrix[index] = np.inf
    # determination of class of training y whose X has i_th min distance
        classy_min_dis = trainy[index_y]
        CLassy_Mins_Dis[i,:] = classy_min_dis
# MAX Vot by sum of class of training y of KNN
    mask = np.sum(CLassy_Mins_Dis,axis=0) > 0
    y_hat = np.zeros_like(mask,dtype=int)
    y_hat[mask] = 1
    y_hat[~mask] = -1
    y_hat = np.reshape(y_hat,(n,1))
    error = np.array(np.where(testy != y_hat)).shape[1]/n * 100
    return y_hat,error
