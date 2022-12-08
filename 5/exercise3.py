from matplotlib import pyplot as plt
import cv2
import a5_utils;
import numpy as np
import math
from PIL import Image

#Task A --------------------------------------------------------------------------------------------------------------

def triangulate(correspondences, P1, P2):
    points3D = []
    for x1, y1, x2, y2 in correspondences:
        #Transform homogenous coordinates vector into array form
        pts1 = np.array([[0, -1, y1], [1, 0, -x1], [-y1, x1, 0]])
        pts2 = np.array([[0, -1, y2], [1, 0, -x2], [-y2, x2, 0]])

        #Calculate the Ax matrices (Product of [x]P)
        A1 = np.dot(pts1, P1)
        A2 = np.dot(pts2, P2)

        #Combine the first 2 lines of A1 and A2 into A matrix
        A = np.append(A1[:2], A2[:2], axis=0)

        #Calculate the SVD of A
        U, D, VT = np.linalg.svd(A)

        #Get the eigen vector (last column of V = the last row of VT) that has the lowest eigenvalue (last value of diagonal matrix D)
        p3d = VT[-1]

        #Normalize the point and remove the extra 1
        p3d = (p3d / p3d[-1])[:3]

        # Add the point to all other 3d points
        points3D.append(p3d)

    return np.array(points3D)






def task_a():
    corr = np.loadtxt("5\data\epipolar\house_points.txt")
    P1 = np.loadtxt("5\data\epipolar\house1_camera.txt")
    P2 = np.loadtxt("5\data\epipolar\house2_camera.txt")
    I1 = np.asarray(Image.open("5\data\epipolar\house1.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("5\data\epipolar\house2.jpg").convert("L")).astype(np.float64) / 255

    points3D = triangulate(corr, P1, P2)

    #Dot product between points3D and T for plotting
    T = np.array([[-1,0,0], [0,0,1], [0,-1,0]])

    points3D = np.dot(points3D, T)

    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(0.3))

    # First subplot
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(I1, cmap="gray")
    ax.plot(corr[:,0], corr[:,1], 'r.', markersize=5)

    # Second subplot
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(I2, cmap="gray")
    ax.plot(corr[:,2], corr[:,3], 'r.', markersize=5)

    # Third subplot (3d)
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot(points3D[:,0],points3D[:,1],points3D[:,2],  "r.", markersize=5)

    plt.show()


task_a()