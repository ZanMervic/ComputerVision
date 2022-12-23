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

def preparation(path="data/faces/1"):
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
    image = cv2.cvtColor(cv2.imread("data/faces/1/001.png"), cv2.COLOR_RGB2GRAY)
    
    #IMAGE 1
    #Convert the image to a vector
    X_1 = np.reshape(np.copy(image), (-1,1))
    #project the points in the PCA space
    X_1 = np.dot(U.T, X_1 - mean)
    #We project the points back to the original space
    X_1 = np.dot(U, X_1) + mean
    image_1 = np.reshape(X_1, image.shape)

    #IMAGE 2
    #Convert the image to a vector
    X_2 = np.reshape(np.copy(image), (-1,1))
    #Change 1 compontnt to 0
    X_2[4074] = 0
    #project the points in the PCA space
    X_2 = np.dot(U.T, X_2 - mean)
    #We project the points back to the original space
    X_2 = np.dot(U, X_2) + mean
    image_2 = np.reshape(X_2, image.shape)

    #IMAGE 3
    #Convert the image to a vector
    X_3 = np.reshape(np.copy(image), (-1,1))
    #project the points in the PCA space
    X_3 = np.dot(U.T, X_3 - mean)
    #change one of its componenets while in PCA space
    X_3[4] = 0
    #We project the points back to the original space
    X_3 = np.dot(U, X_3) + mean
    image_3 = np.reshape(X_3, image.shape)
    

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
    plt.title(f'1 component set to 0 and converted to PCA and back')
    plt.subplot(3, 3, 6)
    plt.imshow(image - image_2)

    plt.subplot(3, 3, 7)
    plt.imshow(image, cmap="gray")
    plt.title(f'Original image')
    plt.subplot(3, 3, 8)
    plt.imshow(image_3, cmap="gray")
    plt.title(f'Converted to PCA had 1 component set to 0 and back')
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
    image = cv2.cvtColor(cv2.imread("data/faces/1/001.png"), cv2.COLOR_RGB2GRAY)
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
# task_c()