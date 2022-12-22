from matplotlib import pyplot as plt
import a6_utils
import numpy as np
from PIL import Image
import cv2

# Task A ----------------------------------------------------------------------------------------------------	

def dualPca(X):
    #We get the matrix shape
    m, N = X.shape
    #We calculate the mean value of the data (seperately for x and y values)
    mean = 1/N * np.sum(X, axis=1,keepdims=True)
    #We center the data using the mean value (from all x/y we substract the mean x/y)
    X_d = X - mean
    #Compute the covariance matrix
    C = 1/(N-1) * np.dot(X_d.T, X_d)
    #Compute the singular value decomposition
    U, S, VT = np.linalg.svd(C)
    #Compute the basis of the eigenvector space ???
    U = X_d @ U @ np.diag(np.sqrt(1/(S*(N-1))))
    return C, mean, U, S, VT

def task_a():
    #Read the points file and reshape it so it becomes a 2xN matrix
    X = np.loadtxt("6\data\points.txt").reshape(-1,2).T
    #Get the covariance matrix, the mean value, the eigenvectors and the eigenvalues
    C, mean, U, S, _ = dualPca(X)
    U

#------------------------------------------------------------------------------------------------------------






# Task B ----------------------------------------------------------------------------------------------------

def task_b():
    #Read the points file and reshape it so it becomes a 2xN matrix
    X = np.loadtxt("6\data\points.txt").reshape(-1,2).T
    #Get the covariance matrix, the mean value, the eigenvectors and the eigenvalues
    C, mean, U, S, _ = dualPca(X)

    #project the points in the PCA space
    X_new = np.dot(U.T, X - mean)
    #We project the points back to the original space
    X_new = np.dot(U, X_new) + mean

    #We plot the original points and the projected points to see if they match
    plt.figure(figsize=(10,6))
    plt.scatter(X[0], X[1])
    plt.scatter(X_new[0], X_new[1], marker='x', color='red')
    plt.show()

#------------------------------------------------------------------------------------------------------------

# task_a()
# task_b()