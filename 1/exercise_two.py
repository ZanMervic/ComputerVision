from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

image = "images/bird.jpg"
bird = "images/bird.jpg"
eagle = "images/eagle.jpg"
coins = "images/coins.jpg"
umbrellas = "images/umbrellas.jpg"

# Creating a binary mask
def task_a():
    I = Image.open(image).convert('L') 
    I = np.asarray(I)
    I = I.astype(np.float64)

    threshold = 53
    mask = np.copy(I)

    # mask[mask<threshold] = 0
    # mask[mask>=threshold] = 1

    mask = np.where(mask<threshold, 0, 1)

    plt.subplot(1,2,1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Binary mask")

    plt.show()

# Creates a histogram from an image and number of bins
def myhist(image, bins):
    pixels = image.reshape(-1).astype(np.int64)
    H = np.zeros(bins).astype(np.uint)
    #Just to make the other tasks faster
    if(bins == 255):
        H = np.bincount(pixels)
    # My own loop
    else:
        for i in pixels:
            index = int(i // (256 / bins))
            H[index] += 1
    return H

# Displaying the histograms
def task_b(numOfBins):
    I = Image.open(image).convert("L")
    I = np.asarray(I) 
    I = I.astype(np.float64)
    heigth, width = I.shape
    H = myhist(I, numOfBins) / (heigth * width)

    plt.subplot(1,2,1)
    plt.imshow(I, cmap="gray")
    
    plt.subplot(1,2,2)
    plt.bar(np.arange(len(H)), H) 

    plt.subplots_adjust(wspace=0.5)
    plt.show()

# Creates a histogram from just an image
def myhist2(image):
    pixels = image.reshape(-1)
    bins = int(max(pixels) - min(pixels))
    H = np.zeros(bins).astype(np.uint)
    for i in pixels:
        index = int(i // (255 / bins)) # 255/bins is the bin range
        H[index] += 1
    return H


# Displays the difference between myhist() and myhist2(). Is there a difference ?
def task_c():
    I = Image.open(image).convert("L")
    I = np.asarray(I) 
    I = I.astype(np.float64)
    H = myhist(I, 255)
    H2 = myhist2(I)

    plt.subplot(1,3,1)
    plt.imshow(I, cmap="gray")
    
    plt.subplot(1,3,2)
    plt.bar(np.arange(len(H)), H)
    plt.title("Old")

    plt.subplot(1,3,3)
    plt.bar(np.arange(len(H2)), H2)
    plt.title("New")

    plt.subplots_adjust(wspace=0.5)
    plt.show()

#Comparing different lighting conditions
def task_d():
    I1 = Image.open("myImages/bright.jpg").convert("L")
    I1 = np.asarray(I1) 
    I1 = I1.astype(np.float64)

    I2 = Image.open("myImages/medium.jpg").convert("L")
    I2 = np.asarray(I2) 
    I2 = I2.astype(np.float64)

    I3 = Image.open("myImages/dark.jpg").convert("L")
    I3 = np.asarray(I3) 
    I3 = I3.astype(np.float64)

    H1 = myhist(I1,255)
    H2 = myhist(I2,255)
    H3 = myhist(I3,255)

    plt.subplot(1,3,1)
    plt.bar(np.arange(len(H1)), H1)
    plt.title("Bright")

    plt.subplot(1,3,2)
    plt.bar(np.arange(len(H2)), H2)
    plt.title("Medium")

    plt.subplot(1,3,3)
    plt.bar(np.arange(len(H3)), H3)
    plt.title("Dark")

    plt.subplots_adjust(wspace=0.5)
    plt.show()


# Expression source: https://www.youtube.com/watch?v=jUUkMaNuHP8
def myOtsu(image):
    heigth, width = image.shape
    num_of_px = heigth * width
    hist = myhist(image, 255)

    max_sigma = 0
    best_threshold = 0
    for threshold in range(0,255):
        if hist[threshold:].sum() <= 0 or hist[0:threshold].sum() <= 0:
            continue

        # We calculate the class probabilities (probability of a pixel having lower/higher value than the threshold)
        w0 = hist[0:threshold].sum() / num_of_px
        w1 = hist[threshold:].sum() / num_of_px

        # We calculate the mean values for each class (class of pixels with lower/higher value that the threshold)
        mean0 = 0
        for i in range(0,threshold):
            mean0 += hist[i] * i
        mean0 = mean0 / hist[0:threshold].sum()
        
        mean1 = 0
        for i in range(threshold,len(hist)):
            mean1 += hist[i] * i
        mean1 = mean1 / hist[threshold:].sum()            

        # We calculate the class variance
        sigma = w0 * w1 * (mean0 - mean1) ** 2

        # The higher the class variance the better the threshold is
        if(sigma > max_sigma):
            max_sigma = sigma
            best_threshold = threshold

    return best_threshold

# Calculating thresholds with otsu's algorithm for different images
def task_e():
    I_bird = Image.open(bird).convert("L")
    I_bird = np.asarray(I_bird) 
    I_bird = I_bird.astype(np.float64)
    print("Bird: %d" % myOtsu(I_bird))

    I_eagle = Image.open(eagle).convert("L")
    I_eagle = np.asarray(I_eagle) 
    I_eagle = I_eagle.astype(np.float64)
    print("Eagle: %d" % myOtsu(I_eagle))

    I_coins = Image.open(coins).convert("L")
    I_coins = np.asarray(I_coins) 
    I_coins = I_coins.astype(np.float64)
    print("Coins: %d" % myOtsu(I_coins))

    I_umbrellas = Image.open(umbrellas).convert("L")
    I_umbrellas = np.asarray(I_umbrellas) 
    I_umbrellas = I_umbrellas.astype(np.float64)
    print("Umbrellas: %d" % myOtsu(I_umbrellas))

# task_a()
# task_b(20)
# task_c()
task_d()
# task_e()