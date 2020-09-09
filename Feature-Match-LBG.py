import numpy as np
import math


def EuclideanDistance(m1, m2):
  """Takes in two matrices as parameters (m1, m2)
  and returns the Euclidean Distance Matrix between them
  """
  
  r = np.shape(m1)[0] #Number of rows in the Euclidean Distance Matrix will be number of rows in m1
  c = np.shape(m2)[0] ##Number of columns in the Euclidean Distance Matrix will be number of rows in m2
 
  EDM = np.empty((r, c)) #Euclidean Distance Matrix EDM (empty) created
 
 
  for i in range(r):
    for j in range(c):
      EDM[i][j] = math.sqrt(np.sum(np.square(np.subtract(m1[i], m2[j]))))
 
  return EDM


def lbg_codebook(fvs, M):
  """
  Uses the LBG Algorithm to generate the codebook using the features
  """
  
  #INITIALIZATION
  no_centroids = 1 #No. of codevectors is 1 intially, so only one centroid
  codebook = np.mean(fvs, axis = 0) #Codebook is the Centroid of features
  e = 0.01  #Splitting factor e = 0.01
  distortion = 1
  r = no_centroids   #number of rows in codebook (initially)
  c = len(codebook)   #number of columns in codebook (initially)
  
  #DOUBLING THE CODEBOOK according to the formula y(n)+ = y(n)*(1+e) , y(n)- = y(n)*(1-e)
  while no_centroids < M: 
    newcodebook = np.empty((2*r, c)) #Creating a temporary codebook that is to be updated
    if no_centroids == 1:
            newcodebook[0] = codebook*(1+e)
            newcodebook[1] = codebook*(1-e)
    else:
      for i in range(no_centroids):
        newcodebook[2*i] = codebook[i] * (1+e)    #y(n)+ = y(n)*(1+e)
        newcodebook[2*i+1] = codebook[i] * (1-e)  #y(n)- = y(n)*(1-e) 

    codebook = newcodebook #codebook updated
    r = np.shape(codebook)[0] #updating the value of centroid // again, the number of centroids = number of codevectors
    no_centroids = r #Again, the number of centroids is the number of rows
  
    Distance = EuclideanDistance(fvs, codebook) #Distance Matrix

    while np.abs(distortion) > e: 
      #NEAREST NEIGHBOUR SEARCH
      previousDistance = np.mean(Distance)
      nearestcodebookID = np.argmin(Distance,axis = 1)  #Contains the indices of the closest codebook for each feature
 
      #SET NEW CENTROID TO THE CENTROID OF ALL FEATURES CLOSE TO CENTROID i
      for i in range(no_centroids):
        codebook[i] = np.mean(fvs[np.where(nearestcodebookID == i)], axis = 0)
 
      no_codebook = np.nan_to_num(codebook) #Replace all non-number values with 0
      
      Distance = EuclideanDistance(fvs, codebook)
      newDistance = np.mean(Distance)
      distortion = (previousDistance - newDistance)/previousDistance  #updating distortion
  return codebook 
