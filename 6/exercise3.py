from os import listdir
from matplotlib import pyplot as plt
import matplotlib.animation
import numpy as np
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






# Task d -----------------------------------------------------------------------------------------------------
#explanation https://discord.com/channels/435483352011767820/626489670787923972/1056989938807230494
def task_d():
    images = preparation("data/faces/2")
    C, mean, U, S, VT = dualPca(images)

    #We get the average photo, which we get by using the mean() function on our images
    avg_img = np.reshape(images.mean(axis=1), (-1,1))
    plt.imshow(avg_img.reshape((96,84)), cmap="gray")
    plt.show()

    #We project this average face to PCA space
    avg_pca = np.dot(U.T, avg_img - mean)
    
    #We look how the 2.eigenvector looks
    snd_egn = U[:,1]
    plt.imshow(snd_egn.reshape((96,84)), cmap="gray")
    plt.show()

    #We create a range from around -10 to 10 and define the scailing factor
    range = np.linspace(-3*np.pi, 3*np.pi, 200)
    scale = 3000

    fig, ax = plt.subplots()
    
    def anim_func(i):
        #For each range we multiply the eigenvector with the range and multiply with the scare to exagerate the results
        avg_pca[1] = np.sin(range[i]) * scale
        #We convert the face back to image space
        avg_pca_img = np.dot(U, avg_pca) + mean
        #We plot the image
        ax.imshow(avg_pca_img.reshape((96, 84)), cmap="gray", vmin=0, vmax=255)


    #Because the eigenvector for which we are changing the weight looks as it looks, we can see one side of the face 
    #getting darker when we apply more and more weight of that vector
    #In PCA space, our image gets converted to an array of 64, that represents the weights for our 64 eigenvectors
    #the first few vectors represent broader features, the latter represent the details
    animation = matplotlib.animation.FuncAnimation(fig, anim_func, frames=len(range))
    animation.resume()
    plt.show()

# -------------------------------------------------------------------------------------------------------------------------







#Task e -------------------------------------------------------------------------------------------------------------------

def task_e():
    images = preparation("data/faces/1")
    #We use the face images in the dualPca function
    C, mean, U, S, VT = dualPca(images)

    #We load the elephant image
    image = cv2.cvtColor(cv2.imread("data/elephant.jpg"), cv2.COLOR_RGB2GRAY)
    X = np.reshape(image, (-1,1))

    #We send the elephant image to the PCA space built by face images and back
    X_pca = np.dot(U.T, X - mean)
    #We project the points back to the original space
    X = np.dot(U, X_pca) + mean
    #It will try to build the elephant image back using only the eigenvectors from the faces which is not possible

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(X, (96,84)), cmap="gray")
    plt.show()

# -------------------------------------------------------------------------------------------------------------------------






#Task g -------------------------------------------------------------------------------------------------------------------

#X input images in columns, arranged by classes
#c number of classes
#n number of images per class
def lda(X, c, n):
    SB = np.zeros((X.shape[0], X.shape[0])) # between class scatter
    SW = np.zeros((X.shape[0], X.shape[0])) # within class scatter
    MM = np.mean(X, axis=0)
    Ms = np.zeros((X.shape[0], c))
    for i in range(1, c):
        Ms[:,i] = np.mean(X[:, (i-1)*n+1 : i*n], axis=1)
        SB = SB + n * np.dot((Ms[:,[i]] - MM),(Ms[:,[i]] - MM).T)
        for j in range(1,n):
            SW = SW + np.dot((X[:, [i*n+j]] - Ms[:,[i]]), (X[:, [i*n+j]] - Ms[:,[i]]).T)

    #https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    W, V = np.linalg.eig(np.linalg.inv(SW) @ SB)
    return np.dot(W, Ms), MM

def task_g():
    images1 = preparation("data/faces/1")
    images2 = preparation("data/faces/2")
    images3 = preparation("data/faces/3")
    combined = np.concatenate((images1, images2, images3), axis=1) / 256

    C, mean, U, S, VT = dualPca(combined)
    combined_pca = np.dot(U.T, combined - mean)
    # plt.scatter(combined_pca[0], combined_pca[1], c=['r']*64 + ['g']*64 + ['b']*64 )
    # plt.show()

    X = combined_pca[:30]
    V, MM = lda(X, 3, 64)
    combined_lda = np.dot(V.T, (X-MM))

    colors = ['c']*64 + ['m']*64 + ['y']*64

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.scatter(-combined_pca[0], -combined_pca[1], c=colors )
    plt.subplot(1,2,2)
    plt.scatter(-combined_lda[0], -combined_lda[1], c=colors )
    plt.show()

# task_g()


# task_b()
# task_c()
# task_d()
# task_e()