from os import listdir
from matplotlib import pyplot as plt
import a6_utils
import numpy as np
from PIL import Image
import cv2


# From previous exercises ----------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------------------------






# Task A ----------------------------------------------------------------------------------------------------

def preparation(path="6/data/faces/1"):
    img_names = listdir(path)
    images = []

    for image in img_names:
        images.append(cv2.cvtColor(cv2.imread(f'{path}/{image}'), cv2.COLOR_RGB2GRAY).flatten())

    return np.asarray(images).T

# ------------------------------------------------------------------------------------------------------------







# Task B ----------------------------------------------------------------------------------------------------

def task_b():
    #Get the images matrix
    images = preparation()
    #Use the dualPCA on the matrix
    C, mean, U, S, VT = dualPca(images)

    #Plot the first 5 eigenvectors
    plt.figure(figsize=(10,3))
    
    #What we see are the first 5 most important eigenvectors, when we want to construct a face,
    #we will combine some of these vectors along with some other to get the face we want
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(np.reshape(U[:,i], (96, 84)), cmap="gray")
    plt.show()

    #We get the image of a face
    image = cv2.cvtColor(cv2.imread("6/data/faces/1/001.png"), cv2.COLOR_RGB2GRAY)
    #Convert the image to a vector
    X = np.reshape(image, (-1,1))
    
    #project the points in the PCA space
    X_pca = np.dot(U.T, X - mean)
    #We project the points back to the original space
    X_new = np.dot(U, X_pca) + mean

    #We get the first image which was just converted to PCA space and back
    image_1 = np.reshape(X_new, image.shape)

    #We get the second image which was converted to PCA space and back + had 1 component set to 0
    image_2 = np.copy(X_new)
    image_2[4074] = 0
    image_2 = np.reshape(image_2, image.shape)

    #We get the third image while changing the vector while in PCA space
    X_pca[4] = 0
    #Convert the changed vector back from PCA space
    X_new = np.dot(U, X_pca) + mean
    image_3 = np.reshape(X_new, image.shape)

    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f'Original image')
    plt.subplot(3, 3, 2)
    plt.imshow(image_1, cmap="gray")
    plt.title(f'Converted to PCA and back')
    plt.subplot(3,3,3)
    plt.imshow(image - image_1)

    plt.subplot(3, 3, 4)
    plt.imshow(image, cmap="gray")
    plt.title(f'Original image')
    plt.subplot(3, 3, 5)
    plt.imshow(image_2, cmap="gray")
    plt.title(f'Converted to PCA and back (1 component set to 0)')
    plt.subplot(3, 3, 6)
    plt.imshow(image - image_2)

    plt.subplot(3, 3, 7)
    plt.imshow(image, cmap="gray")
    plt.title(f'Original image')
    plt.subplot(3, 3, 8)
    plt.imshow(image_3, cmap="gray")
    plt.title(f'Converted to PCA (1 component set to 0) and back')
    plt.subplot(3, 3, 9)
    plt.imshow(image - image_3)


    plt.show()


# ------------------------------------------------------------------------------------------------------------







# Task C-------------------------------------------------------------------------------------------------------


def task_c():
    #Get the images matrix
    images = preparation()
    #Use the dualPCA on the matrix
    C, mean, U, S, VT = dualPca(images)

    #We get the image of a face
    image = cv2.cvtColor(cv2.imread("6/data/faces/1/001.png"), cv2.COLOR_RGB2GRAY)
    #Convert the image to a vector
    X = np.reshape(image, (-1,1))

    plt.figure(figsize=(10,6))
    nums = [32,16,8,4,2,1]
    for i, num in np.ndenumerate(nums):
        #project the points in the PCA space
        X_pca = np.dot(U.T, X - mean)
        #Remove the component with the lowest eigenvalue
        newU = np.copy(U)
        newU[:,num:] = 0
        #We project the points back to the original space
        X_new = np.dot(newU, X_pca) + mean
        img = np.reshape(X_new, image.shape)
        plt.subplot(1,6, i[0]+1)
        plt.imshow(img, cmap="gray")
    plt.show()


# ------------------------------------------------------------------------------------------------------------


task_b()
task_c()