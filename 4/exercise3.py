from matplotlib import pyplot as plt
import cv2
import a4_utils;
import numpy as np
import math
from PIL import Image

from exercise2 import find_matches


#TASK A-------------------------------------------------------------------------------------------------------------------

#As an input we get a arrays of coordinates of correspondences
#xr - array of x cordinates of first image
#yr - array of y cordinates of first image
#xt - array of x cordinates of second image
#yt - array of y cordinates of second image
def estimate_homography(correspondences):
    A = []
    #We construct the matrix A using the "formula" in the instructions
    #For each xr,yr,xt,yt we add 2 rows to the matrix A
    for xr, yr, xt, yt in correspondences:
        A.append([xr, yr, 1, 0, 0, 0, -xt * xr, -xt * yr, -xt])
        A.append([0, 0, 0, xr, yr, 1, -yt * xr, -yt * yr, -yt])
    #Perform a matrix decomposition using the SVD algorithm
    #VT is always of length 9x9 ?
    U, S, VT = np.linalg.svd(A)
    #We calculate the vector h using the formula -> 9x1 matrix
    h = VT[-1] / VT[-1,-1]
    #Reshape h to a 3x3 matrix H
    H = np.reshape(h, (3,3))
    return H


def task_a():
    triplets = [["data/newyork/newyork_a.jpg","data/newyork/newyork_b.jpg","data/newyork/newyork.txt"], ["data\graf\graf_a.jpg", "data\graf\graf_b.jpg", "data\graf\graf.txt"]]
    for image1, image2, data in triplets:
        I1 = np.asarray(Image.open(image1).convert("L")).astype(np.float64) / 255
        I2 = np.asarray(Image.open(image2).convert("L")).astype(np.float64) / 255
        #We get the correspondences from a txt file
        correspondences = np.loadtxt(data)
        #Store correspondences of the first image into points1 and second into points2
        points1 = correspondences[:,:2]
        points2 = correspondences[:,2:]
        #Displaying the matching points using the given methon
        a4_utils.display_matches(I1, points1, I2, points2)

        #Calculating the homography matrix (it tells us the rotation of the image based on the correspondences ?)
        H = estimate_homography(correspondences)
        #Transform the first image to the plane of the second image
        plane = cv2.warpPerspective(I1, H, dsize=I1.shape)

        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.imshow(I1)
        plt.title("Original")
        plt.subplot(2,1,2)
        plt.imshow(plane)
        plt.title("Wrap")
        plt.show()


#-------------------------------------------------------------------------------------------------------------------








#TASK B-------------------------------------------------------------------------------------------------------------------
def ransac(correspondences, samples, error, iterations):
    #Here we will sotre the largest number of inliers and the corresponding model
    maxM = 0
    maxH = []
    #We start the ransac loop
    for i in range(0, iterations):
        #Select 4 random correpondences
        np.random.shuffle(correspondences)
        randCorr = correspondences[:samples]
        #Calculate the H for the random correspondences
        H = estimate_homography(randCorr)
        points1 = correspondences[:,:2]
        points2 = correspondences[:,2:]
        #Multiply the points coordinates with the H and calculating the error
        #We take the points of the first image, convert them to a 1x3 vector (by adding 1) -> [x,y,1]T
        points1 = np.pad(points1, ((0,0),(0,1)), "constant", constant_values=(1))
        points1 = np.expand_dims(points1,2) #Used to transpose all the vectors
        #We multiply these vectors with the H matrix and divide the result by the last vector value so we again get 1 in that position -> [x',y',1]
        points1_ = H.dot(points1) #We do the dot product between the H matrix and each vector
        points1_ = np.squeeze(points1_.T) #We transform the result into a (num of points)x3 array
        points1_ = points1_ / np.expand_dims(points1_[:,-1],1) #We divide each row by it's last element
        points1_ = points1_.astype(int) #Convert the points back to integers
        points1_ = np.delete(points1_,2,1) #We delete the last column (the colum with 1s)
        #We conpute the distance between these vectors and their corresponding matches and count how many fall within the allowed error
        distances = np.sqrt((points1_[:,0] - points2[:,0])**2 + (points1_[:,1] - points2[:,1])**2)
        M = np.where(distances < error)[0].size #We count the amount of points whose distances are lower than the allowed error
        #If the amount of valid points is higher than the current amount of valid points, we keep the new homography matrix because it's better than the old one
        if(M > maxM):
            maxM = M
            maxH = H
    return maxH
        
        


def task_b():
    I1 = np.asarray(Image.open("data/newyork/newyork_a.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("data/newyork/newyork_b.jpg").convert("L")).astype(np.float64) / 255

    points1, points2 = find_matches(I1, I2, 1)
    a4_utils.display_matches(I1, points1, I2, points2)


    correspondences = np.hstack((points1,points2))
    H = ransac(correspondences, 4, 2, 100)

    plane = cv2.warpPerspective(I1, H, dsize=I1.shape)

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.imshow(I2)
    plt.title("newyork_b")
    plt.subplot(2,1,2)
    plt.imshow(plane)
    plt.title("transformed newyork_a")
    plt.show()

#-------------------------------------------------------------------------------------------------------------------