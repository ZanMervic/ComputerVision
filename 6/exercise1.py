from matplotlib import pyplot as plt
import a6_utils
import numpy as np


# Task A ----------------------------------------------------------------------------------------------------
# Use this: https://matrixcalc.org/vectors.html#eigenvectors(%7B%7B4%2e25,0%2e33%7D,%7B0%2e33,1%2e33%7D%7D)
# --------------------------------------------------------------------------------------------------------







# Task B ----------------------------------------------------------------------------------------------------

def pca(X):
    #We get the matrix shape
    m, N = X.shape
    #We calculate the mean value of the data (seperately for x and y values)
    mean = 1/N * np.sum(X, axis=1,keepdims=True)
    #We center the data using the mean value (from all x/y we substract the mean x/y)
    X_d = X - mean
    #Compute the covariance matrix
    C = 1/(N-1) * np.dot(X_d, X_d.T)
    #Compute the eigenvalues and eigenvectors of the covariance matrix by using the SVD
    U, S, VT = np.linalg.svd(C)
    #COLUMNS of U are the eigenvectors
    #values of S are the eigenvalues
    #We return all of this because we need it for the next tasks
    return C, mean, U, S, VT

def task_b():
    #Read the points file
    X = np.loadtxt("data\points.txt")
    #Reshape it so it becomes a 2xN matrix
    X = X.reshape(-1,2).T
    C, mean, _, _, _ = pca(X)
    #We plot the points and draw the elipse
    plt.figure(figsize=(10,6))
    plt.scatter(X[0], X[1])
    a6_utils.drawEllipse(mean, C, n_std=1)
    plt.show()

#------------------------------------------------------------------------------------------------------------






# Task C ----------------------------------------------------------------------------------------------------
def task_c():
    #Read the points file and reshape it so it becomes a 2xN matrix
    X = np.loadtxt("data\points.txt").reshape(-1,2).T
    #Get the covariance matrix, the mean value, the eigenvectors and the eigenvalues
    C, mean, U, S, _ = pca(X)
    #We plot the points and draw the elipse
    plt.figure(figsize=(10,6))
    plt.scatter(X[0], X[1])
    a6_utils.drawEllipse(mean, C, n_std=1)
    #We square root the eigenvalues (otherwise it doesn't work)
    S = np.sqrt(S)
    #We plot the eigenvectors sclaled by the eigenvalues
    plt.plot([mean[0], mean[0] + U[0,0] * S[0]], [mean[1], mean[1] + U[1,0] * S[0]], color='red')
    plt.plot([mean[0], mean[0] + U[0,1] * S[1]], [mean[1], mean[1] + U[1,1] * S[1]], color='green')
    plt.show()

# -----------------------------------------------------------------------------------------------------------





# Task D ----------------------------------------------------------------------------------------------------


def task_d():
    #Read the points file and reshape it so it becomes a 2xN matrix
    X = np.loadtxt("data\points.txt").reshape(-1,2).T
    #Get the covariance matrix, the mean value, the eigenvectors and the eigenvalues
    _, _, _, S, _ = pca(X)
    #We plot the eigenvalues using the bar plot
    plt.figure(figsize=(10,6))
    #We plot the cumulative sum of the eigenvalues (we divide by the sum of the eigenvalues so we normalize it)
    #Cumulative sum (cumsum) of array [1,2,3] is [1,3,6], of array [1,1,1] is [1,2,3]
    #So the second array element is the sum of the first and itself, the third is the sum of the first, second and itself and so on
    #The last element is the sum of all the elements, that's why we divide by the sum of the eigenvalues to normalize it
    plt.bar([1,2], np.cumsum(S)/np.sum(S))
    plt.show()


# -----------------------------------------------------------------------------------------------------------






# Task E ----------------------------------------------------------------------------------------------------

def task_e():
    #Read the points file and reshape it so it becomes a 2xN matrix
    X = np.loadtxt("data\points.txt").reshape(-1,2).T
    #Get the covariance matrix, the mean value, the eigenvectors and the eigenvalues
    C, mean, U, S, _ = pca(X)

    #project the points in the PCA space
    X_new = np.dot(U.T, X - mean)
    #We remove the components from U that have the lowest eigenvalue
    newU = np.copy(U)
    newU[:,-1] = 0
    #We project the points back to the original space
    X_new = np.dot(newU, X_new) + mean

    #We plot the original points, the projected points, the elipse and the eigenvectors (in that order)
    plt.figure(figsize=(10,6))
    plt.scatter(X[0], X[1])
    plt.scatter(X_new[0], X_new[1], marker='x', color='red')
    a6_utils.drawEllipse(mean, C, n_std=1)
    S = np.sqrt(S)
    plt.plot([mean[0], mean[0] + U[0,0] * S[0]], [mean[1], mean[1] + U[1,0] * S[0]], color='red')
    plt.plot([mean[0], mean[0] + U[0,1] * S[1]], [mean[1], mean[1] + U[1,1] * S[1]], color='green')
    plt.show()


# -----------------------------------------------------------------------------------------------------------







# Task F ----------------------------------------------------------------------------------------------------


def task_f():
    #Read the points file and reshape it so it becomes a 2xN matrix
    X = np.loadtxt("data\points.txt").reshape(-1,2).T
    #Get the covariance matrix, the mean value, the eigenvectors and the eigenvalues
    C, mean, U, S, _ = pca(X)

    #New points q = [6,6]T
    q = np.array([[6],[6]])

    #Calculate the closest point to q using euclidean distance
    #1.calculate all the distances
    distances = np.sqrt((q[0][0] - X[0,:])**2 + (q[1][0] - X[1,:])**2)
    #2.find the index of the minimum distance
    index = np.where(distances == min(distances))[0][0]
    #3.get the point with that index
    closest_point = X[:,index]

    #Project the points to the PCA space
    X_new = np.dot(U.T, X - mean)
    q_new = np.dot(U.T, q - mean)
    #Remove the component with the lowest eigenvalue
    newU = np.copy(U)
    newU[:,-1] = 0
    #We project the points back to the original space
    X_new = np.dot(newU, X_new) + mean
    q_new = np.dot(newU, q_new) + mean

    #We calculate the distances again and find the new closest point
    distances_new = np.sqrt((q_new[0][0] - X_new[0,:])**2 + (q_new[1][0] - X_new[1,:])**2)
    index_new = np.where(distances_new == min(distances_new))[0][0]
    closest_point_new = X_new[:,index_new]

    #We plot the original points, the projected points, the elipse and the eigenvectors (in that order)
    plt.figure(figsize=(10,6))
    plt.scatter(X[0], X[1])
    plt.scatter(X_new[0], X_new[1], marker='x', color='red')
    a6_utils.drawEllipse(mean, C, n_std=1)
    S = np.sqrt(S)
    plt.plot([mean[0], mean[0] + U[0,0] * S[0]], [mean[1], mean[1] + U[1,0] * S[0]], color='red')
    plt.plot([mean[0], mean[0] + U[0,1] * S[1]], [mean[1], mean[1] + U[1,1] * S[1]], color='green')
    plt.scatter(q[0], q[1], marker='+', color='black')
    plt.scatter(closest_point[0], closest_point[1], marker='+', color='black')
    plt.scatter(q_new[0], q_new[1], marker='+', color='purple')
    plt.scatter(closest_point_new[0], closest_point_new[1], marker='+', color='purple')
    plt.show()
    #Black + are the point q and it's closest point in the original space
    #Purple + are the point q and it's closest point after the transformation


# -----------------------------------------------------------------------------------------------------------

# task_b()
# task_c()
# task_d()
# task_e()
# task_f()






