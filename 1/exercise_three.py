from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from exercise_two import myOtsu

bird = "images/bird.jpg"
eagle = "images/eagle.jpg"
coins = "images/coins.jpg"

# Creates a (inverted) mask from an image an threshold
def mask(I, threshold, inverse=False):
    #threshold 53 works best for bird
    if(inverse):
        I = np.where(I>threshold, 0, 1)
    else:
        I = np.where(I<threshold, 0, 1)
    return I 

# Playing with SE and morphological operations
def task_a():
    I = Image.open(bird).convert('L')
    I = np.asarray(I) #type=uint8
    I = I.astype(np.float64) #type=float64
    

    I = mask(I, 53) 
    I = I.astype('uint8')

    n = 5
    # create a square structuring element
    SE = np.ones((n,n), np.uint8) 

    #"I" must be of type uint8
    I_eroded = cv2.erode(I, SE)
    I_dilated = cv2.dilate(I, SE)

    I_closed = cv2.erode(I_dilated, SE) #Erosion after dilation
    I_opened = cv2.dilate(I_eroded, SE) #Dilation after erotion

    plt.subplot(1,5,1)
    plt.title("Original")
    plt.imshow(I)

    plt.subplot(1,5,2)
    plt.title("Eroded")
    plt.imshow(I_eroded)

    plt.subplot(1,5,3)
    plt.title("Dilated")
    plt.imshow(I_dilated)

    plt.subplot(1,5,4)
    plt.title("Opened")
    plt.imshow(I_opened)

    plt.subplot(1,5,5)
    plt.title("Closed")
    plt.imshow(I_closed)

    plt.show()

# Cleaning up the bird.jpg mask 
def task_b():
    I = Image.open(bird).convert('L')
    I = np.asarray(I)
    I = I.astype(np.float64)
    I = mask(I, 53)
    I = I.astype('uint8')

    n = 6
    # create a square structuring element
    SE = np.ones((n,n), np.uint8) 
    #create an ellipse SE
    SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n)) 

    I2 = cv2.dilate(I, SE)
    I2 = cv2.dilate(I2, SE)
    I2 = cv2.erode(I2, SE)
    I2 = cv2.dilate(I2, SE)
    I2 = cv2.erode(I2, SE)

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(I, cmap="gray")

    plt.subplot(1,2,2)
    plt.title("New")
    plt.imshow(I2, cmap="gray")

    plt.show()


#Task c -------------------------------------------------------------
def immask(I, M):
    # for idx, x in np.ndenumerate(I):
    #     I[idx] = I[idx] * M[idx[0], idx[1]]

    I[M == 0] = 0

    return I


def task_c():
    I = Image.open(bird).convert('RGB')  # PIL image.
    I = np.asarray(I)  # Converting to Numpy array.
    I = I.astype(np.float64) / 255

    I2 = Image.open(bird).convert('L')
    I2 = np.asarray(I2)
    M = mask(I2, 53)

    I = immask(I,M)

    plt.imshow(I)
    plt.show()


# Using Otsu and immask on eagle.jpg
def task_d():
    I = Image.open(eagle).convert('RGB')  # PIL image.
    I = np.asarray(I)  # Converting to Numpy array.
    I = I.astype(np.float64) / 255

    M = Image.open(eagle).convert('L')
    M = np.asarray(M)
    #Otsu seth the threshold at 126 (better at 180)
    M = mask(M, 180) #If we did mask(1 - M, 180) the object would be included in the mask
    

    IM = immask(I,M)

    plt.imshow(IM)
    plt.show()

# Removing large coins using masks, morphological operations and connected components
def task_e():
    I = Image.open(coins).convert("L")
    I = np.asarray(I)

    M = mask(I, 250, inverse=True)
    M = M.astype(np.uint8)

    n = 15
    n2 = 5
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))
    SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n2,n2))

    M = cv2.erode(M, SE)
    M = cv2.dilate(M, SE)
    M = cv2.erode(M, SE2)
    

    # In the labeled_image each coin has a different pixel color/value from 0 (background) to num_coins - 1
    _ , labeled_image, stats, _ = cv2.connectedComponentsWithStats(M)

    # We keep only the sizes row of stats
    sizes = stats[:, -1]
    max_area = 700

    #Here we remove all coins whose area is larger than 700px
    for i in range(len(sizes)):
        if(sizes[i] > max_area):
            # Where the labeled image has a pixel value of i set the pixel value of M to 0
            M[labeled_image == i] = 0


    original = Image.open(coins).convert("RGB")
    original = np.asarray(original)

    plt.subplot(1,4,1)
    plt.imshow(original)
    plt.title("Original")

    plt.subplot(1,4,2)
    plt.imshow(M, cmap="gray")
    plt.title("Small coins only mask")

    plt.subplot(1,4,3)
    plt.imshow(immask(original, M))
    plt.title("Small coins only on black")

    # We remove the large coins from the image
    original[M == 0] = 255

    plt.subplot(1,4,4)
    plt.imshow(original)
    plt.title("Small coins only")

    plt.show()


# task_a()
# task_b()
# task_c()
# task_d()
# task_e()