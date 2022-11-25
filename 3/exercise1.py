from matplotlib import pyplot as plt
import cv2
import a3_utils;
import numpy as np
import math
from PIL import Image



#FROM ASSIGNMENT2-----------------------------------------------------------------------------
def gaussian_kernel(sigma, w=0):
    w = 2 * math.ceil(sigma*3) + 1 if w == 0 else w
    x = np.linspace(-math.ceil(sigma*3) - 0.5, math.ceil(sigma*3) + 0.5, w) #Create a numpy array with kernel_shape values from - to + kernel_shape
    kernel = (1 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-(x**2)/(2*sigma**2)) #We fill the array with gaussian_kernel values (Put each value from the x into this equasion and store the result)
    kernel = kernel / sum(kernel) #Normalize the kernel
    return np.flip(kernel.reshape((-1,1)).T) #Convert to a 2d array (So we can later use .T and some other methods on it)
#---------------------------------------------------------------------------------------------





#TASK A-------------------------------------------Na listu---------------------------------------------------------------






#TASK B-------------------------------------------------------------------------------------------------------------------
def gaussdx(sigma, w=0):
    w = 2 * math.ceil(sigma*3) + 1 if w == 0 else w
    x = np.linspace(-math.ceil(sigma*3) - 0.5, math.ceil(sigma*3) + 0.5, w) #Create a numpy array with kernel_shape values from - to + kernel_shape
    kernel = -(1 / (np.sqrt(2*np.pi)*(sigma**3))) * x * np.exp(-(x**2)/(2*sigma**2)) #We fill the array with gaussian_kernel values (Put each value from the x into this equasion and store the result)
    kernel = kernel / sum(np.abs(kernel)) #Normalize the kernel
    return np.flip(kernel.reshape((-1,1)).T) #Convert to a 2d array (So we can later use .T and some other methods on it)
#-------------------------------------------------------------------------------------------------------------------







#TASK C-------------------------------------------------------------------------------------------------------------------
def task_c():
    G = gaussian_kernel(4)
    D = gaussdx(4)
    
    impulse = np.zeros((50,50))
    impulse[25,25] = 255
    plt.figure(figsize=(10,6))

    plt.subplot(2,3,1)
    plt.imshow(impulse, cmap="gray")
    plt.title("Impulse")
    
    plt.subplot(2,3,2)
    plt.imshow(cv2.filter2D(src=cv2.filter2D(src=impulse, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T), cmap="gray")
    plt.title("G,Dt")

    plt.subplot(2,3,3)
    plt.imshow(cv2.filter2D(src=cv2.filter2D(src=impulse, ddepth=-1, kernel=D), ddepth=-1, kernel=G.T), cmap="gray")
    plt.title("D,Gt")

    plt.subplot(2,3,4)
    plt.imshow(cv2.filter2D(src=cv2.filter2D(src=impulse, ddepth=-1, kernel=G), ddepth=-1, kernel=G.T), cmap="gray")
    plt.title("G,Gt")

    plt.subplot(2,3,5)
    plt.imshow(cv2.filter2D(src=cv2.filter2D(src=impulse, ddepth=-1, kernel=G.T), ddepth=-1, kernel=D), cmap="gray")
    plt.title("Gt,D")

    plt.subplot(2,3,6)
    plt.imshow(cv2.filter2D(src=cv2.filter2D(src=impulse, ddepth=-1, kernel=D.T), ddepth=-1, kernel=G), cmap="gray")
    plt.title("Dt,G")

    plt.show()
#-------------------------------------------------------------------------------------------------------------------

    






#TASK D-------------------------------------------------------------------------------------------------------------------

def partial(I,G,D):
    Ix = cv2.filter2D(src=cv2.filter2D(src=I, ddepth=-1, kernel=G.T), ddepth=-1, kernel=D)
    Iy = cv2.filter2D(src=cv2.filter2D(src=I, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    return Ix, Iy

def second(Iy,Ix,G,D):
    Ixx = cv2.filter2D(src=cv2.filter2D(src=Ix, ddepth=-1, kernel=G.T), ddepth=-1, kernel=D)
    Ixy = cv2.filter2D(src=cv2.filter2D(src=Ix, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    Iyy = cv2.filter2D(src=cv2.filter2D(src=Iy, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    return Ixx, Ixy, Iyy

def gradient_magnitude(I, sigma=1):
    G = gaussian_kernel(1)
    D = gaussdx(1)
    Ix, Iy = partial(I,G,D)
    Imag = np.sqrt(Ix**2 + Iy**2)
    Idir = np.arctan2(Iy,Ix,)
    return Imag, Idir


def task_d(image = "images\museum.jpg"):
    I = np.asarray(Image.open(image).convert("L")).astype(np.float64)

    G = gaussian_kernel(1)
    D = gaussdx(1)

    Ix, Iy = partial(I,G,D)
    Ixx, Ixy, Iyy = second(Iy, Ix, G, D)
    Imag, Idir = gradient_magnitude(I)

    plt.figure(figsize=(10,6))
    plt.subplot(2,4,1)
    plt.imshow(I, cmap="gray")
    plt.subplot(2,4,2)
    plt.imshow(Ix, cmap="gray")
    plt.subplot(2,4,3)
    plt.imshow(Iy, cmap="gray")
    plt.subplot(2,4,4)
    plt.imshow(Imag, cmap="gray")
    plt.subplot(2,4,5)
    plt.imshow(Ixx, cmap="gray")
    plt.subplot(2,4,6)
    plt.imshow(Ixy, cmap="gray")
    plt.subplot(2,4,7)
    plt.imshow(Iyy, cmap="gray")
    plt.subplot(2,4,8)
    plt.imshow(Idir, cmap="gray")

    plt.show()
#-------------------------------------------------------------------------------------------------------------------


# task_c()
# task_d()
