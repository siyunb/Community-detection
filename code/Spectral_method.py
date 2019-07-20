# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:44:20 2019

@author: Bokkin Wang
"""
import sys
sys.path.append("D:/bigdatahw/pan_paper/simulation")
import numpy as np
from sklearn.cluster import KMeans
from math import log
from sklearn import datasets
import warnings
from matplotlib import pyplot as plt
from itertools import cycle, islice
warnings.filterwarnings("ignore")

####visualization
def plot(X, y_km, y_sp, y_sc):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_km) + 1))))
    plt.subplot(131)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")

    plt.subplot(132)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    
    plt.subplot(133)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_sc])
    plt.title("Score Clustering")

####distance matrix
def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

####create adjacency matrix
def adjacency_matrix(S, k, sigma=1.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
    return A

####construct laplacian matrix
def calLaplacianMatrix(adjacentMatrix, mode = 'NCut'):
    # compute the Degree of adjacency matrix : D
    degreeMatrix = np.sum(adjacentMatrix, axis=1)
    # construct the laplacian matrix : L
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix    
    #RatioCut: utilize the Laplacian matrix directly
    if mode == 'Ratiocut':
        return laplacianMatrix    
    #Ncut: normalize the L to D^(-1/2) L D^(-1/2)
    else:
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

####calculate eigenvalues and eigenvectors of laplacian matrix 
def caleigen(Laplacian):
    lam, V =  np.linalg.eig(Laplacian) 
    lam = zip(lam, range(len(lam)))
    lam = sorted(lam, key=lambda lam:lam[0])
    H = np.vstack([V[:,i] for (v, i) in lam]).T
    lam = [item[0] for item in lam]
    return lam, H

####simulation data
def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y

## SCORE ##########################################################################
def Score(A_matrix, K_num):
    #set parameter
    n = len(A_matrix)
    T_n = log(n) #threshold
    
    #calculate eigen    
    eigen_values, eigen_vectors = caleigen(A_matrix) # the vector's been normalized
    g_1 = eigen_vectors[:,0]
    
    #Calculate R_star
    R_star = np.zeros((len(A_matrix), K_num-1))
    for i in range(K_num-1):
        R_star[:,i] = eigen_vectors[:,i+1]/g_1
    R_star[R_star > T_n] = T_n    
    R_star[R_star < -T_n] = -T_n
              
    # K-means step
    sc_kmeans = KMeans(n_clusters = K_num).fit(R_star)
    return sc_kmeans


if __name__ == "__main__": 
    np.random.seed(1)
    data, label = genTwoCircles(n_samples=500)
    Similarity = calEuclidDistanceMatrix(data)
    Adjacent = adjacency_matrix(Similarity, k=10)
    Laplacian = calLaplacianMatrix(Adjacent)
    lam, H = caleigen(Laplacian)
    sp_kmeans = KMeans(n_clusters=2).fit(H[...,0:2])
    pure_kmeans = KMeans(n_clusters=2).fit(data)
    sc_kmeans= Score(Laplacian, 2)
    plot(data, pure_kmeans.labels_, sp_kmeans.labels_, sc_kmeans.labels_)









