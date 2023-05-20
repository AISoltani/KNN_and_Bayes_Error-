import numpy as np
from data_gen import data_gen
from scipy.spatial.distance import cdist
from knn import knn
n = 6000
d = 2
sigma = 1
b = 0
w = np.zeros([1,d-1])
w = np.insert(w,0,[1])
w = np.reshape(w,(1,d))
k_list=[1,3,5]
dist = 1 # dist norm => if dist = 1, it means 1 norm

trainX,trainy =  data_gen (n , d, sigma, w, b)
testX,testy =  data_gen (n , d, sigma, w, b)
for k in k_list:
    y_hat,error = knn(trainX, trainy, testX, k, dist, testy)
    print("k ",k,"Error %: ", error)
############ data-gen functin
# import numpy as np
# def  data_gen (n , d, sigma, w, b):
#     X = np.random.uniform(-1, 1, (n, d))
#     U = np.random.uniform(0, 1, (1, n))
#     B = b * np.ones([n,1])
    
#     SGM = 1/(1 + np.exp(-(np.dot(w,X.T) + B.T)/sigma))
#     mask =  U <= SGM 
#     y = np.zeros_like(mask,dtype=int)
#     y[mask] = 1
#     y[~mask] = -1
#     return X,y

# import numpy as np
# from data_gen import data_gen
# from scipy.spatial.distance import cdist

########### knn function
# def knn(trainX, trainy, testX, k, dist, testy):
#     trainy = np.reshape (trainy,-1)
#     testy = testy.T
#     n = trainX.shape[0]
# # distance of text data from the trained data
# # Each row is the distance of each test data from all of the trained data
#     Dist_Matrix = cdist(testX,trainX, metric='minkowski', p= dist)
#     CLassy_Mins_Dis = np.zeros([k,n])
#     for i in range(k):
#     # Each element of following vector (MIN) is 
#     # the min of each test data from trained data is determined
#         MIN = np.min(Dist_Matrix,axis=1)
#         MIN = np.reshape(MIN,(n,1)) # change the row vector to column vector
#     # the index of those mins 
#         index = np.where(Dist_Matrix == MIN) 
#         index_x,index_y = index
#     # change the mins to inf because if k>1 we need other mins 
#     # forexample ( 3rd mins 5th mins and etc.)
#         Dist_Matrix[index] = np.inf
#     # determination of class of training y whose X has i_th min distance
#         classy_min_dis = trainy[index_y]
#         CLassy_Mins_Dis[i,:] = classy_min_dis
# # MAX Vot by sum of class of training y of KNN
#     mask = np.sum(CLassy_Mins_Dis,axis=0) > 0
#     y_hat = np.zeros_like(mask,dtype=int)
#     y_hat[mask] = 1
#     y_hat[~mask] = -1
#     y_hat = np.reshape(y_hat,(n,1))
#     error = np.array(np.where(testy != y_hat)).shape[1]/n * 100
#     return y_hat,error