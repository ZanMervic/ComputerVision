from matplotlib import pyplot as plt
import cv2
import a2_utils;
from PIL import Image
import numpy as np

from exercise2 import gaussian_kernel
from exercise2 import convolution


def gauss_filter(image, sigma=2):
    kernel = gaussian_kernel(sigma)[::-1] #Generate a 1D kernel and flip it
    image = cv2.filter2D(src=image,ddepth=-1,kernel=kernel) #Appying convolution using the 1D kernel
    image = cv2.filter2D(src=image,ddepth=-1,kernel=kernel[np.newaxis]) #Transposing the kernel and applying convolution again
    return image

def task_a():
    I = np.asarray(Image.open("./images/lena.png").convert("L")).astype(np.float64)
    I_gauss = a2_utils.gauss_noise(I, magnitude=15)
    I_sp = a2_utils.sp_noise(I)

    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.subplot(2,3,2)
    plt.imshow(I_gauss, cmap="gray")
    plt.title("Gaussian Noise")
    plt.subplot(2,3,3)
    plt.imshow(I_sp, cmap="gray")
    plt.title("Sale and Pepper")
    plt.subplot(2,3,5)
    plt.imshow(gauss_filter(I_gauss), cmap="gray")
    plt.title("Filtered Gaussian noise")
    plt.subplot(2,3,6)
    plt.imshow(gauss_filter(I_sp), cmap="gray")
    plt.title("Filtered Salt and pepper")

    plt.show()

def task_b():
    #Reading the image with np.asarray(Image.open("./images/museum.jpg").convert("L")).astype(np.float64) doesn't work
    I = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY)

    #Creating the same kernel as in the lecture slides
    g1 = np.asarray([[0,0,0],[0,2,0],[0,0,0]])
    g2 = np.asarray([[1,1,1],[1,1,1],[1,1,1]]) * 1/9
    kernel = g1 - g2
    
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(I, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(cv2.filter2D(src=I, ddepth=-1, kernel=kernel), cmap="gray")

    plt.show()


#The median filter does not need to be of width 2N + 1
def simple_median(I, w):
    I2 = np.zeros(I.shape)

    for i in range(len(I)):
        #Some simple start and end indexes, so we don't get out of the array
        #and we are always around/on the index with the kernel
        start = max(0, i+1-w)
        end = min(start+w, len(I))
        I2[i] = np.median(I[start:end])
    return I2

def task_c():
    #Creating the signal
    signal = np.zeros(40)
    signal[6:20] = 1

    #Creating the corrupted signal
    corrupted = signal.copy()
    corrupted[np.random.rand(corrupted.shape[0]) < 0.1 / 2] = 3
    corrupted[np.random.rand(corrupted.shape[0]) < 0.1 / 2] = 0

    plt.figure(figsize=(10,6))
    plt.subplot(1,4,1)
    plt.plot(signal)
    plt.ylim([0,6])
    plt.title("Original")
    plt.subplot(1,4,2)
    plt.plot(corrupted)
    plt.ylim([0,6])
    plt.title("Corrupted")
    plt.subplot(1,4,3)
    #Filtering with gauss
    plt.plot(convolution(corrupted, gaussian_kernel(2)))
    plt.ylim([0,6])
    plt.title("Gauss")
    plt.subplot(1,4,4)
    #Filtering with a median filter
    plt.plot(simple_median(corrupted,3))
    plt.ylim([0,6])
    plt.title("Median")


    plt.show()

#For padding reasons i will assume the filter width is 2N + 1
def simple_median_2d(I,w):
    new_I = np.zeros_like(I) #Creating empty array of size I
    heigth, width = new_I.shape

    N = (w - 1) // 2
    I = np.pad(I, N) #Adding padding to I (zeros)

    #Iterating through the array and calculating medians
    for j in range(heigth):
        for i in range(width):
            new_I[j,i] = np.median(I[j:j+w-1,i:i+w-1])

    return new_I

def task_d():
    I = cv2.cvtColor(cv2.imread("./images/lena.png"), cv2.COLOR_BGR2GRAY)
    I_gauss = a2_utils.gauss_noise(I, 25)
    I_sp = a2_utils.sp_noise(I)

    plt.figure(figsize=(10,6))

    plt.subplot(2,4,1)
    plt.imshow(I,cmap="gray")
    plt.title("Original")
    plt.subplot(2,4,2)
    plt.imshow(I_gauss,cmap="gray")
    plt.title("Gaussian noise")
    plt.subplot(2,4,3)
    plt.imshow(gauss_filter(I_gauss,2),cmap="gray")
    plt.title("Gauss filtered")
    plt.subplot(2,4,4)
    plt.imshow(simple_median_2d(I_gauss, 5),cmap="gray")
    plt.title("Median filtered")
    plt.subplot(2,4,6)
    plt.imshow(I_sp,cmap="gray")
    plt.title("Salt and pepper")
    plt.subplot(2,4,7)
    plt.imshow(gauss_filter(I_sp,2),cmap="gray")
    plt.title("Gauss filtered")
    plt.subplot(2,4,8)
    plt.imshow(simple_median_2d(I_sp, 5),cmap="gray")
    plt.title("Median filtered")

    plt.show()


def laplacian_filter(image, sigma):
    #Generate a 1D gauss kernel
    gauss_kernel = gaussian_kernel(sigma) 

    #Creating the laplacian kernel
    laplacian_kernel = np.zeros_like(gauss_kernel)
    laplacian_kernel[laplacian_kernel.size // 2] = 1
    laplacian_kernel = laplacian_kernel - gauss_kernel
    laplacian_kernel = laplacian_kernel[::-1]

    image = cv2.filter2D(src=image,ddepth=-1,kernel=laplacian_kernel) #Appying convolution using the 1D kernel
    image = cv2.filter2D(src=image,ddepth=-1,kernel=laplacian_kernel[np.newaxis]) #Transposing the kernel and applying convolution again

    return image

def task_e():
    I1 = np.asarray(Image.open("./images/lincoln.jpg").convert("L")).astype(np.float64)
    I2 = np.asarray(Image.open("./images/obama.jpg").convert("L")).astype(np.float64)

    I_gauss = gauss_filter(I1, 5) #Odstrani visoke frekvence (kjer se intensity hitro spremeni)
    I_laplace = laplacian_filter(I2, 25) #Odstrani niske frekvence (kjer se intensity pocasi spreminja)

    I_result = (I_gauss * 0.6) + (I_laplace * 0.8)

    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.imshow(I1, cmap="gray")
    plt.title("image 1")
    plt.subplot(2,3,2)
    plt.imshow(I2, cmap="gray")
    plt.title("image 2")
    plt.subplot(2,3,3)
    plt.imshow(I_result, cmap="gray")
    plt.title("Result")
    plt.subplot(2,3,4)
    plt.imshow(I_gauss, cmap="gray")
    plt.title("Gauss")
    plt.subplot(2,3,5)
    plt.imshow(I_laplace, cmap="gray")
    plt.title("Laplace")

    plt.show()


# task_a()
# task_b()
# task_c()
task_d()
# task_e()


