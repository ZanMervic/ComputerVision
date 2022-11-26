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
    plt.imshow(I2, cmap="gray")
    plt.title("newyork_b")
    plt.subplot(2,1,2)
    plt.imshow(plane, cmap="gray")
    plt.title("transformed newyork_a")
    plt.show()

#-------------------------------------------------------------------------------------------------------------------
#Help on discord: https://discord.com/channels/435483352011767820/626489670787923972/1046011888498114621

#The idea here was to "transform" each pixel from the original image to the warped image using the H (homography) matrix
#But because after the dot product between H and our pixel coordinates we get a float and need to round it to get a matrix index
#Multiple pixels sometimes get mapped to the same index in the new matrix (leaving some pixels missing -> artifacts)

#My idea was that for every black pixel in the new image we would inverse the coordinates and fetch the value from the original image
#to fill in the black artifacts (did not work very well, but i could see a cleaner image in the corners)
#Thats when i got the idea to delete the first part of the algorythm and just use the "reverse homography"
def warpPerspectiveAttempt1(I, H):
    Iwarp = np.zeros_like(I) #Here we will store the result
    inv_H = np.linalg.inv(H) #We inverse the H matrix because otherwise it did not work
    #For each pixel in I we do:
    for (x, y), value in np.ndenumerate(I):
        x_r = [x, y, 1] #Transform the pixel coordinates to homogenous form
        x_t = inv_H.dot(x_r) #Compute the corresponding x_t using the homography matrix
        x_t = np.round(x_t / x_t[-1]).astype(int) #Transform the x_t back to 2D space
        #Because we are rotating the image not all pixels will be placed in the new image (some will get cut off)
        if(x_t[0] >= 0 and x_t[0] < Iwarp.shape[1] and x_t[1] >= 0 and x_t[1] < Iwarp.shape[0]):
            Iwarp[x_t[0], x_t[1]] = value

    #The result currently has many artifacts
    #For every black pixel in the new image we do a reverse homography and fetch the value of the real image
    for (x,y), value in np.ndenumerate(Iwarp):
        if(value != 0):
            continue
        x_r = [x, y, 1] #Transform the pixel coordinates to homogenous form
        x_t = inv_H.dot(x_r) #Compute the corresponding x_t using the homography matrix
        x_t = np.round(x_t / x_t[-1]).astype(int) #Transform the x_t back to 2D space
        #Because we are rotating the image not all pixels will be placed in the new image (some will get cut off)
        if(x_t[0] >= 0 and x_t[0] < Iwarp.shape[1] and x_t[1] >= 0 and x_t[1] < Iwarp.shape[0]):
            Iwarp[y, x] = I[x_t[1], x_t[0]]
        


    return Iwarp

#Instead of transforming the reference image coordinates and copying the values to the new image (and loosing some pixels because of rounding floats)
#we iterate through each pixel in the new image, do a inverse transform to get the reference image coordinates and fetching the corresponding value
#We can still get some distortion (nearby pixels fetch the same reference image value) but we don't have any black pixels
def inverseWrapPerspective(I, H):
    Iwarp = np.zeros_like(I) #Here we will store the result
    inv_H = np.linalg.inv(H) #We inverse the H matrix because otherwise it did not work (tried to transpose it, did not work)
    #For each pixel in the new image we will transform the coordinates to the reference image (using inv(H))
    #And store the value of the reference image into our new image
    for y in range(0, Iwarp.shape[0]):
        for x in range(0, Iwarp.shape[1]):
            new_homog_coords = [x, y, 1] #Transform the pixel coordinates to homogenous form
            ref_homog_coords = inv_H.dot(new_homog_coords) #Compute the corresponding ref_homog_coords using the homography matrix
            ref_homog_coords = np.round(ref_homog_coords / ref_homog_coords[-1]).astype(int) #Transform the ref_homog_coords back to 2D space
            #Because we are rotating the image not all pixels will be placed in the reference image (some will get cut off)
            if(ref_homog_coords[0] >= 0 and ref_homog_coords[0] < Iwarp.shape[1] and ref_homog_coords[1] >= 0 and ref_homog_coords[1] < Iwarp.shape[0]):
                Iwarp[y, x] = I[ref_homog_coords[1], ref_homog_coords[0]]

    return Iwarp


def task_d():
    I1 = np.asarray(Image.open("data/newyork/newyork_a.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("data/newyork/newyork_b.jpg").convert("L")).astype(np.float64) / 255

    points1, points2 = find_matches(I1, I2, 1)
    correspondences = np.hstack((points1,points2))
    H = ransac(correspondences, 4, 2, 100)

    plane1 = warpPerspectiveAttempt1(I1, H)
    plane2 = inverseWrapPerspective(I1, H)


    plt.figure(figsize=(10,6))
    plt.subplot(2,3,2)
    plt.imshow(I2, cmap="gray")
    plt.title("newyork_b")
    plt.subplot(2,3,4)
    plt.imshow(plane1, cmap="gray")
    plt.title("transformed newyork_a - attempt 1")
    plt.subplot(2,3,6)
    plt.imshow(plane2, cmap="gray")
    plt.title("transformed newyork_a - inverse")
    plt.show()

task_d()