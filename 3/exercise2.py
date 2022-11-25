from matplotlib import pyplot as plt
import cv2
import a3_utils;
import numpy as np
import math
from PIL import Image
from exercise1 import gaussdx
from exercise1 import gaussian_kernel



#-------------------------------------------------------------------------------------------------------------------------

#The I needs to be a nparray (grayscale image)
def mag_and_dir(I, sigma):
    G = gaussian_kernel(sigma)
    D = gaussdx(sigma)

    Ix = cv2.filter2D(src=cv2.filter2D(src=I, ddepth=-1, kernel=G.T), ddepth=-1, kernel=D)
    Iy = cv2.filter2D(src=cv2.filter2D(src=I, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    Imag = np.sqrt(Ix**2 + Iy**2)
    Idir = np.arctan2(Iy,Ix,)
    return(Imag, Idir)

#-------------------------------------------------------------------------------------------------------------------------



#TASK A-------------------------------------------------------------------------------------------------------------------

#If something won't work correctly it might be because filter2D used 0's padding which max create a false edge/line
#I = image, sigma = filter size, theta = threshold
def findedges(I, sigma, theta):
    Imag, _ = mag_and_dir(I, sigma)

    Imag = np.where(Imag<theta, 0, 1)
    edges = Imag
    return edges

def task_a(image = "images\museum.jpg"):
    I = np.asarray(Image.open(image).convert("L")).astype(np.float64) / 255

    Iedges1 = findedges(I, 1, 0.16)
    Iedges2 = findedges(I, 1, 0.20)
    Iedges3 = findedges(I, 1, 0.10)
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(Iedges1, cmap="gray")
    plt.title("Theta = 0.16")
    plt.subplot(1,3,2)
    plt.imshow(Iedges2, cmap="gray")
    plt.title("Theta = 0.20")
    plt.subplot(1,3,3)
    plt.imshow(Iedges3, cmap="gray")
    plt.title("Theta = 0.10")
    plt.show()

#-------------------------------------------------------------------------------------------------------------------









#TASK B-------------------------------------------------------------------------------------------------------------------

#Mogoce uporabi kaksen drug nacin za pretvorit v 1-8
def angle_to_1_to_8(angle):
    #we need to use -angle, dont know why
    temp = -angle + np.pi #instead of -pi to pi we now have 0 to 2pi
    temp = temp / np.pi #instead of 0 to 2pi we have 0 to 2
    temp = temp * 4 #now we have 0 - 8 which is almost what we want
    temp = round(temp) #now we have integers from 0 - 8 ... bingo

    return temp

#Imag = magnitude image, Idir = gradient image
def non_maxima_suppression(Imag, Idir):
    result = np.zeros_like(Imag)
    #We iterate through all the pixels
    for x in range(0,Imag.shape[0]-1):
        for y in range(0, Imag.shape[1]-1):
            pixel = Imag[x,y]
            #We calculate the border gradient direction
            #Each Idir image pixels has a value from -pi to pi which gives us the angle of the gradient
            #We need to convert this angle to numbers from 1 to 8 or 1 to 4 to determine which pixels we need to compare our pixel to
            angle = angle_to_1_to_8(Idir[x,y])

            max_neighbor = 0
            if(angle == 2 or angle == 6):
                # --
                max_neighbor = max(Imag[x+1,y], Imag[x-1,y])
            elif(angle == 1 or angle == 5):
                # /
                max_neighbor = max(Imag[x-1,y+1],Imag[x+1,y-1])
            elif(angle == 0 or angle == 8 or angle == 4):
                # |
                max_neighbor = max(Imag[x,y-1],Imag[x,y+1])
            elif(angle == 3 or angle == 7):
                # \
                max_neighbor = max(Imag[x-1,y-1],Imag[x+1,y+1])
            
            result[x,y] = 0 if pixel < max_neighbor else pixel

    return result

def task_b(image="images\museum.jpg", sigma=1):
    I = np.asarray(Image.open(image).convert("L")).astype(np.float64) / 255
    Imag, Idir = mag_and_dir(I, sigma)

    non_maxima = non_maxima_suppression(Imag, Idir)

    plt.imshow(non_maxima, cmap="gray")
    plt.show()

#-------------------------------------------------------------------------------------------------------------------









#TASK C-------------------------------------------------------------------------------------------------------------------
# I is the result of non-max-suppresion
def hysteresis(I, t_high, t_low):
    I = np.where(I<t_low, 0, I) #Getting rid of all pixels lower than t_low
    Ibin = np.where(I>0, 1,0).astype(np.uint8) #Creating a binary image for connectedComponentsWithStats
    #n - number of different connected components
    #label - labeled image with values from 0 to n
    n , label, _, _ = cv2.connectedComponentsWithStats(Ibin)
    for i in range(1,n):
        if np.max(I[label==i]) > t_high:
            I[label==i] = 1
        else:
            I[label==i] = 0

    return I



def task_c(image="images\museum.jpg", sigma=1):
    I = np.asarray(Image.open(image).convert("L")).astype(np.float64) / 255
    Imag, Idir = mag_and_dir(I, sigma)
    Ithr = findedges(I, 1, 0.16)
    Inonmax = non_maxima_suppression(Imag, Idir)
    Ihyst = hysteresis(Inonmax,0.16, 0.04)

    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.subplot(2,2,2)
    plt.imshow(Ithr, cmap="gray")
    plt.title("Thresholded")
    plt.subplot(2,2,3)
    plt.imshow(np.where(Inonmax > 0.16, 1, 0), cmap="gray")
    plt.title("Nonmax. supp.")
    plt.subplot(2,2,4)
    plt.imshow(Ihyst, cmap="gray")
    plt.title("Hysteresis")

    plt.show()

# task_a()
# task_b()
# task_c()