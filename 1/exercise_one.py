from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

umbrellas = 'images/umbrellas.jpg'

# Displaying an image
def task_a():
    I = imread(umbrellas)
    imshow(I)

# Converting to grayscale
def task_b():
    I = imread(umbrellas)
    height, width, channels = I.shape

    I_Gray = np.zeros((height,width))

    # Looping through every pixel and averaging the values
    for i in np.arange(I.shape[0]):
        for j in np.arange(I.shape[1]):
            I_Gray[i,j] = np.sum(I[i,j]) / 3

    imshow(I_Gray)

# Displaying a cutout
def task_c():
    I = imread(umbrellas)

    cutout = I[130:260, 240:450, 0]

    plt.subplot(1,2,1)
    plt.imshow(I)
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(cutout, cmap="gray")
    plt.title("Red channel cutout")

    plt.show()

# Inverting part of an image
def task_d():
    I = imread(umbrellas)

    #Here we did "1 - I" instead of "255 - I" because of the scailing to [0,1] from [0,255] 
    cutout = 1 - I[130:260, 240:450]

    # for idx, x in np.ndenumerate(cutout):
    #     cutout[idx] = 1 - x 

    # for i in np.arange(cutout.shape[0]):
    #     for j in np.arange(cutout.shape[1]):
    #         for x in range(3):
    #             cutout[i,j,x] = 1 - cutout[i,j,x]

    I[130:260, 240:450] = cutout
    imshow(I)   

# Reduction of grayscale levels (less colors used to display the image)
def task_e():
    I = imread_gray(umbrellas)
    I2 = np.copy(I) * 63
    I2 = I2.astype(np.uint8)

    plt.subplot(1,2,1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(I2,vmax=255, cmap="gray")
    plt.title("Modified")

    plt.show()

# task_a()
# task_b()
# task_c()
# task_d()
# task_e()

