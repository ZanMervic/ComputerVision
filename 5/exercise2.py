from matplotlib import pyplot as plt
import cv2
import a5_utils;
import numpy as np
import math
from PIL import Image


#Task B --------------------------------------------------------------------------------------------------------------


def fundamental_matrix(pts1, pts2):
    #Normalize the points, so that the centroid is at the origin and on average the distance from the origin is sqrt(2), get the transformation matrix.
    pts1, T1 = a5_utils.normalize_points(pts1)
    pts2, T2 = a5_utils.normalize_points(pts2)

    correspondences = np.concatenate((pts1[:,0:2], pts2[:,0:2]), axis=1)

    #Create the matrix A
    A = []
    for u, v, u_, v_ in (correspondences):
        A.append([u*u_, u_*v, u_, u*v_, v*v_, v_, u, v, 1])
    
    A = np.array(A)

    #Compute the SVD of A https://www.youtube.com/watch?v=nbBvuuNVfco
    U, D, V = np.linalg.svd(A)

    #The last column of V (eigenvector v9) is the solution to the equation Av = 0
    #Transforming the last eigenvector v9 in a 3 × 3 fundamental matrix F
    F = V[-1].reshape(3,3)

    #Enforce the rank 2 constraint
    #Decomposing the fundamental matrix F into UDV^T
    U, D, V = np.linalg.svd(F)
    #Setting the smallest eigenvalue to zero
    D[2] = 0
    #Reconstructing the fundamental matrix F
    F = np.dot(U, np.dot(np.diag(D), V))

    #Transforming the fundamental matrix F back to the original coordinate system
    F = np.dot(T2.T, np.dot(F, T1))

    return F

    
def task_b():
    correspondences = np.loadtxt("5\data\epipolar\house_points.txt")
    I1 = np.asarray(Image.open("5\data\epipolar\house1.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("5\data\epipolar\house2.jpg").convert("L")).astype(np.float64) / 255

    #Store correspondences of the first image into pts1 and second into pts2
    pts1 = correspondences[:,:2]
    pts2 = correspondences[:,2:]

    #Compute the fundamental matrix F
    F = fundamental_matrix(pts1, pts2)

    #Transfrom the points into homogeneous coordinates
    h_pts1 = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    h_pts2 = np.hstack((pts2, np.ones((pts2.shape[0],1))))

    #Compute the epipolar lines for each point
    L1 = []
    for row in h_pts1:
        L1.append(np.dot(F, row))
    L2 = []
    for row in h_pts2:
        L2.append(np.dot(F.T, row))

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(I1, cmap="gray")
    #Draw the epipolar lines
    for line in L2:
        a5_utils.draw_epiline(line, I2.shape[0], I2.shape[1])
    #Plot the points
    plt.plot(pts1[:,0], pts1[:,1], 'r.', markersize=10)

    plt.subplot(1,2,2)
    plt.imshow(I2, cmap="gray")
    #Draw the epipolar lines
    for line in L1:
        a5_utils.draw_epiline(line, I1.shape[0], I1.shape[1])
    #Plot the points
    plt.plot(pts2[:,0], pts2[:,1], 'r.', markersize=10)

    plt.show()


#Task C --------------------------------------------------------------------------------------------------------------

