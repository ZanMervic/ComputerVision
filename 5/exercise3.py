from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

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
    corr = np.loadtxt("data\epipolar\house_points.txt")
    P1 = np.loadtxt("data\epipolar\house1_camera.txt")
    P2 = np.loadtxt("data\epipolar\house2_camera.txt")
    I1 = np.asarray(Image.open("data\epipolar\house1.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("data\epipolar\house2.jpg").convert("L")).astype(np.float64) / 255

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

#-------------------------------------------------------------------------------------------------------------------











#TASK B-------------------------------------------------------------------------------------------------------------

#Perform a 3d reconstruction of an object

#https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga687a1ab946686f0d85ae0363b5af1d7b
def task_b():
    #Load the calibration images, for each image we convert it to grayscale and resize it to 20% of its original size
    Ical = []
    for i in range(1, 11):
        Ical.append(cv2.resize(cv2.cvtColor(cv2.imread(f"./calibration/{i}.jpg"), cv2.COLOR_BGR2GRAY), (0, 0), fx=0.2, fy=0.2))
        

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(3,10,0)
    objp = np.zeros((4*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:4].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #Detect circle centers using cv2.findCirclesGrid, the (4, 11) is the number of circles in the x and y direction
    #The flags are used to detect the circles in an asymmetric grid (the circles are not perfectly aligned)
    for i in range(len(Ical)):
        ret, centers = cv2.findCirclesGrid(Ical[i], (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if(ret == True):
            objpoints.append(objp)
            imgpoints.append(centers)
            centers2 = cv2.cornerSubPix(Ical[i],centers, (11,11), (-1,-1), criteria)
            imgpoints.append(centers2)

            #Draw and display the corners (for debugging)
            Ical[i] = cv2.drawChessboardCorners(Ical[i], (4, 11), centers2, ret)
            plt.imshow(Ical[i])
            plt.show()
            

    #Calibrate the camera using the 3d and 2d coordinates
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, Ical[0].shape[::-1], None, None)


    first = cv2.cvtColor(cv2.imread('data/rubics/20221210_132244.jpg'), cv2.COLOR_RGB2GRAY)
    first = cv2.cvtColor(cv2.imread('calibration/1.jpg'), cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(cv2.imread('data/rubics/20221210_132246.jpg'), cv2.COLOR_RGB2GRAY)

    h, w = first.shape[:2]
    newMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    fst = cv2.undistort(first, mtx, dist, None, newMtx)
    # x, y, w, h = roi
    # fst = fst[y:y+h, x:x+w]

    plt.imshow(fst, cmap="gray")
    plt.show()

    sec = cv2.undistort(second, mtx, dist, None, newMtx)

    

task_a()
