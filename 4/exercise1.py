from matplotlib import pyplot as plt
import cv2
import a4_utils;
import numpy as np
import math
from PIL import Image

#TASK A-------------------------------------------------------------------------------------------------------------------

#This function is copied from the assignment3
#thresh -> points lower than it we just ignore
#size -> amount of pixels we check on each side
def nonmaxima_suppression_box(I, thresh, size=1):
    #The idea is similar to the 2d median from assignment2
    result = np.zeros_like(I) #Create a 0s array 
    I = np.pad(I, size) #Pad I with 1 layer of 0s
    for j in range(result.shape[0]):
        for i in range(result.shape[1]):
            if(I[j+size,i+size] < thresh):
                continue
            #If the pixel is higher than 0 we compare it to the pixels around it
            temp = np.copy(I[j:j+(2*size+1),i:i+(2*size+1)])
            temp[size,size] = 0
            max_neighbor = np.max(temp)
            if(I[j+size,i+size] > max_neighbor):
                #If the pixel is higher than all of it's neighbors we keep it
                result[j,i] = I[j+size,i+size]
            elif(I[j+size,i+size] == max_neighbor):
                I[j+size,i+size] = 0

    return result


#A function that will give us all first and second image derivatives (we need them often)
def derivatives(I, sigma):
    G = a4_utils.gauss(sigma)
    D = a4_utils.gaussdx(sigma)
    Ix = cv2.filter2D(src=cv2.filter2D(src=I, ddepth=-1, kernel=G.T), ddepth=-1, kernel=D)
    Iy = cv2.filter2D(src=cv2.filter2D(src=I, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    Ixx = cv2.filter2D(src=cv2.filter2D(src=Ix, ddepth=-1, kernel=G.T), ddepth=-1, kernel=D)
    Iyy = cv2.filter2D(src=cv2.filter2D(src=Iy, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    Ixy = cv2.filter2D(src=cv2.filter2D(src=Ix, ddepth=-1, kernel=G), ddepth=-1, kernel=D.T)
    return Ix,Iy,Ixx,Iyy,Ixy


def hessian_points(I, sigma, thresh=0.004):
    _, _, Ixx, Iyy, Ixy = derivatives(I,sigma)
    #Just following the formula
    H = Ixx * Iyy - Ixy**2
    #We then do a non_max_supp and thresholding on it
    Hmax = nonmaxima_suppression_box(H, thresh)
    #And we return a touple (H is just for displaying), (np.nonzero(Hmax) will give us the coords of feature points)
    return H, np.nonzero(Hmax)

def task_a(image="./data/graf/graf_a.jpg"):
    I = np.asarray(Image.open(image).convert("L")).astype(np.float64) / 255

    plt.figure(figsize=(10,6))
    for i in range(1,4):
        H, fPoints = hessian_points(I, 3*i)

        plt.subplot(2,3,i)
        plt.imshow(H)

        plt.subplot(2,3,i+3)
        plt.imshow(I, cmap="gray")
        plt.scatter(fPoints[1], fPoints[0], color="red", s=1, marker="x", linewidths=5)

    plt.show()

#-------------------------------------------------------------------------------------------------------------------








#TASK B-------------------------------------------------------------------------------------------------------------------

#BAJE MI MANJKA EN MINUS PR KONVOLUCIJAH ?


def harris_points(I, sigma, alpha = 0.06, thresh = 0.000001):
    Ix, Iy, _, _, Ixy = derivatives(I,sigma)
    G = a4_utils.gauss(sigma*1.6)
    #Again, following the formula, C is a 2x2 matrix which elements are Cx, Cy and 2x Cxy
    Cx = cv2.filter2D(src=cv2.filter2D(src=Ix**2, ddepth=-1, kernel=G), ddepth=-1, kernel=G.T)
    Cy = cv2.filter2D(src=cv2.filter2D(src=Iy**2, ddepth=-1, kernel=G), ddepth=-1, kernel=G.T)
    Cxy = cv2.filter2D(src=cv2.filter2D(src=Ix*Iy, ddepth=-1, kernel=G), ddepth=-1, kernel=G.T)

    #Then we calculate the matrix determinant and trace (basic linear algebra)
    detC = Cx * Cy - Cxy**2
    traceC = Cx + Cy

    #And again follow the formula and do a non_max_supp
    C = detC - alpha * traceC**2

    #And we return a touple (C is just for displaying), (np.nonzero(Cmax) will give us the coords of feature points)
    Cmax = nonmaxima_suppression_box(C, thresh)

    return C, np.nonzero(Cmax)

def task_b(image="./data/graf/graf_a.jpg"):
    I = np.asarray(Image.open(image).convert("L")).astype(np.float64) / 255

    plt.figure(figsize=(10,6))
    for i in range(1,4):
        C, fPoints = harris_points(I, i*3)

        plt.subplot(2,3,i)
        plt.imshow(C)

        plt.subplot(2,3,i+3)
        plt.imshow(I)
        plt.scatter(fPoints[1], fPoints[0], color="red", s=1, marker="x", linewidths=5)

    plt.show()

#-------------------------------------------------------------------------------------------------------------------


# task_a()
# task_b()