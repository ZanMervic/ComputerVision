from matplotlib import pyplot as plt
import cv2
import a2_utils;
import numpy as np
import math

def simple_convolution(I, k):
    I = np.array(I)
    k = np.array(k[::-1]) #Flipping the kernel
    offset = (len(k)-1) // 2 #Calculating the offset used at the end of the I

    H = []
    #The reason for this range is: we put the first element of the kernel on the index
    #so when we start the offset is automaticaly applied because of the shape of the kernel
    #we need to stop at 2*offset because that is where the start of the kernel will be when 
    #the end of the kernel will be at the end of the image.
    for i in range(0, len(I) - 2 * offset):
        H.append(np.sum(I[i:i + len(k)] * k))

    return H

def convolution(I, k):
    I = np.array(I)
    k = np.array(k[::-1]) #Flipping the kernel
    N = (len(k) - 1) // 2
    padding = np.zeros(N)
    I = np.concatenate([padding, I, padding])

    H = []
    for i in range(0, len(I) - 2*N):
        H.append(np.sum(I[i:i + len(k)] * k))

    return H

def task_b():
    signal = a2_utils.read_data("signal.txt")
    kernel = a2_utils.read_data("kernel.txt")
    result = simple_convolution(signal, kernel)
    cv2_result = cv2.filter2D(src=signal, ddepth=-1,kernel=kernel)

    plt.figure(figsize=(10,6))
    plt.plot(signal, label="signal")
    plt.plot(kernel, label="kernel")
    plt.plot(result, label="result")
    plt.plot(cv2_result, label="cv2")
    plt.legend()
    plt.show()


def task_c():
    signal = a2_utils.read_data("signal.txt")
    kernel = a2_utils.read_data("kernel.txt")
    result = convolution(signal, kernel)
    cv2_result = cv2.filter2D(src=signal, ddepth=-1,kernel=kernel)

    plt.figure(figsize=(10,6))
    plt.plot(signal, label="signal")
    plt.plot(kernel, label="kernel")
    plt.plot(result, label="result")
    plt.plot(cv2_result, label="cv2", linestyle="dashed")
    plt.legend()
    plt.show()


def gaussian_kernel_tuple(sigma):
    x = np.linspace(-math.ceil(sigma*3) - 0.5, math.ceil(sigma*3) + 0.5, 2 * math.ceil(sigma*3) + 1) #Create a numpy array with kernel_shape values from - to + kernel_shape
    kernel = (1 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-(x**2)/(2*sigma**2)) #We fill the array with gaussian_kernel values (Put each value from the x into this equasion and store the result)
    kernel = kernel / sum(kernel) #Normalize the kernel
    return kernel, x #We need to return the x for plotting

def gaussian_kernel(sigma):
    x = np.linspace(-math.ceil(sigma*3) - 0.5, math.ceil(sigma*3) + 0.5, 2 * math.ceil(sigma*3) + 1) #Create a numpy array with kernel_shape values from - to + kernel_shape
    kernel = (1 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-(x**2)/(2*sigma**2)) #We fill the array with gaussian_kernel values (Put each value from the x into this equasion and store the result)
    kernel = kernel / sum(kernel) #Normalize the kernel
    return kernel

def task_d():
    plt.figure(figsize=(10,6))

    sigmas = [0.5,1,2,3,4]
    for sigma in sigmas:
        kernel, x = gaussian_kernel_tuple(sigma)
        plt.plot(x, kernel, label=f'sigma={sigma}')

    plt.legend()
    plt.show()

def task_e():
    signal = a2_utils.read_data("signal.txt")
    k1 = np.asarray(gaussian_kernel(2))
    k2 = np.asarray([0.1,0.6,0.4])
    k3 = convolution(k1,k2)

    plt.figure(figsize=(10,3))
    plt.subplot(1,4,1)
    plt.plot(signal)
    plt.title("s")
    plt.subplot(1,4,2)
    plt.plot(convolution(convolution(signal, k1), k2))
    plt.title("(s * k1) * k2")
    plt.subplot(1,4,3)
    plt.plot(convolution(convolution(signal, k2), k1))
    plt.title("(s * k2) * k1")
    plt.subplot(1,4,4)
    plt.plot(convolution(signal,k3))
    plt.title("s * (k1 * k2)")

    plt.show()

# task_b()
# task_c()
# task_d()
# task_e()


