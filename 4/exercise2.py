from matplotlib import pyplot as plt
import cv2
import a4_utils;
import numpy as np
import math
from PIL import Image

from exercise1 import harris_points
from exercise1 import hessian_points



#TASK A-------------------------------------------------------------------------------------------------------------------

def find_correspondences(desc1, desc2):
    #For each descriptor from the first image we...
    pairs = []
    for i,desc in enumerate(desc1):
        #calculate the hellinger distance between it and all descriptors from the second image
        #(axis=1 tells the sum function to only sum elements on the axis 1 -> gives us a 1d array of sums -> we get an array of distances and not just 1 distance)
        dist = np.sqrt(np.sum((np.sqrt(desc) - np.sqrt(desc2))**2, axis=1) / 2)
        #we store the indexes of dest1 and dest2 descriptors whose distance was the smallest
        pairs.append([i,np.where(dist == dist.min())[0][0]])

    return pairs

def task_a():
    I1 = np.asarray(Image.open("data\graf\graf_a_small.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("data\graf\graf_b_small.jpg").convert("L")).astype(np.float64) / 255

    #We get the feature points using the hessian detector
    
    sigma = 2
    _, fPoints1 = hessian_points(I1, sigma)
    _, fPoints2 = hessian_points(I2, sigma)

    # sigma = 5
    # _, fPoints1 = harris_points(I1, sigma)
    # _, fPoints2 = harris_points(I2, sigma)

    #We get the descriptors from out feature points
    desc1 = a4_utils.simple_descriptors(I1, fPoints1[0], fPoints1[1], radius=30, sigma=2)
    desc2 = a4_utils.simple_descriptors(I2, fPoints2[0], fPoints2[1], radius=30, sigma=2)

    #we get a list of pairs which represent indexes in fPoints arrays 
    #So for example pairs[0] = [x1,x2] means that the first pair of points is 
    #[fPoints1[1][x1], fPoints1[0][x1]] and [fPoints2[1][x2], fPoints2[0][x2]]
    correspondences = find_correspondences(desc1, desc2)

    points1 = []
    points2 = []
    for x1, x2 in correspondences:
        points1.append([fPoints1[1][x1], fPoints1[0][x1]])
        points2.append([fPoints2[1][x2], fPoints2[0][x2]])

    a4_utils.display_matches(I1, points1, I2, points2)

#-------------------------------------------------------------------------------------------------------------------







#TASK B-------------------------------------------------------------------------------------------------------------------


def find_correspondences_symetric(desc1, desc2):
    #For each descriptor from the first image we...
    firstPairs = []
    for i,desc in enumerate(desc1):
        #calculate the hellinger distance between it and all descriptors from the second image
        #(axis=1 tells the sum function to only sum elements on the axis 1 -> gives us a 1d array of sums -> we get an array of distances and not just 1 distance)
        dist = np.sqrt(np.sum((np.sqrt(desc) - np.sqrt(desc2))**2, axis=1) / 2)
        #we store the indexes of dest1 and dest2 descriptors whose distance was the smallest
        firstPairs.append([i,np.where(dist == dist.min())[0][0]])

    #Same for the second image we ...
    secondPairs = []
    for i,desc in enumerate(desc2):
        #calculate the hellinger distance between it and all descriptors from the first image
        #(axis=1 tells the sum function to only sum elements on the axis 1 -> gives us a 1d array of sums -> we get an array of distances and not just 1 distance)
        dist = np.sqrt(np.sum((np.sqrt(desc) - np.sqrt(desc1))**2, axis=1) / 2)
        #we store the indexes of dest1 and dest2 descriptors whose distance was the smallest
        secondPairs.append([i,np.where(dist == dist.min())[0][0]])

    #We now have pairs from first and second image
    firstPairs = np.asarray(firstPairs)
    secondPairs = np.flip(secondPairs, axis=1)
    #We convert these arrays to sets, and keep only the matching pairs
    pairs = np.array([x for x in set(tuple(x) for x in firstPairs) & set(tuple(x) for x in secondPairs)])
    return pairs


def find_matches(I1, I2, sigma, detector="hessian", radius=30, descSigma = 2):

    if(detector == "hessian"):
        _, fPoints1 = hessian_points(I1, sigma)
        _, fPoints2 = hessian_points(I2, sigma)
    else:
        _, fPoints1 = harris_points(I1, sigma)
        _, fPoints2 = harris_points(I2, sigma)

    #We get the descriptors from out feature points
    desc1 = a4_utils.simple_descriptors(I1, fPoints1[0], fPoints1[1], radius=radius, sigma=descSigma)
    desc2 = a4_utils.simple_descriptors(I2, fPoints2[0], fPoints2[1], radius=radius, sigma=descSigma)

    #We find symetric correspondences (desc1 matched with desc2 and the other way round)
    correspondences = find_correspondences_symetric(desc1, desc2)

    points1 = []
    points2 = []
    for x1, x2 in correspondences:
        points1.append([fPoints1[1][x1], fPoints1[0][x1]])
        points2.append([fPoints2[1][x2], fPoints2[0][x2]])

    #We return the matching points for each image
    return points1, points2


def task_b():
    I1 = np.asarray(Image.open("data\graf\graf_a_small.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("data\graf\graf_b_small.jpg").convert("L")).astype(np.float64) / 255

    points1, points2 = find_matches(I1, I2, 1)


    a4_utils.display_matches(I1, points1, I2, points2)

#-------------------------------------------------------------------------------------------------------------------






#I used the idea from the lectures, where we only keep those correspondences, which are very similar to eachother
#but not so similar to any other points, which ensures us, that we don't use any correspondences for which we are
#not sure to which point does it translate to. (Full description = presentation 5, slide 88)
#TASK C-------------------------------------------------------------------------------------------------------------------

def find_correspondences_symetric_snd_most_similar(desc1, desc2):
    #For each descriptor from the first image we...
    firstPairs = []
    for i,desc in enumerate(desc1):
        #calculate the hellinger distance between it and all descriptors from the second image
        #(axis=1 tells the sum function to only sum elements on the axis 1 -> gives us a 1d array of sums -> we get an array of distances and not just 1 distance)
        dist = np.sqrt(np.sum((np.sqrt(desc) - np.sqrt(desc2))**2, axis=1) / 2)
        #we get the 2 smallest distances of descriptors (2 most similar descriptors)
        fst_min_dist, snd_min_dist = np.partition(dist, 1)[:2]
        #We check if the ration of these 2 distances is less than 0.8 (the smaller the ratio, the more distinctive the key-point is, which is what we want)
        if(fst_min_dist/snd_min_dist < 0.8):
            #we store the indexes of dest1 and dest2 descriptors whose distance was the smallest
            firstPairs.append([i,np.where(dist == dist.min())[0][0]])

    #Same for the second image we ...
    secondPairs = []
    for i,desc in enumerate(desc2):
        #calculate the hellinger distance between it and all descriptors from the first image
        #(axis=1 tells the sum function to only sum elements on the axis 1 -> gives us a 1d array of sums -> we get an array of distances and not just 1 distance)
        dist = np.sqrt(np.sum((np.sqrt(desc) - np.sqrt(desc1))**2, axis=1) / 2)
        #we get the 2 smallest distances of descriptors (2 most similar descriptors)
        fst_min_dist, snd_min_dist = np.partition(dist, 1)[:2]
        #We check if the ration of these 2 distances is less than 0.8 (the smaller the ratio, the more distinctive the key-point is, which is what we want)
        if(fst_min_dist/snd_min_dist < 0.8):
            #we store the indexes of dest1 and dest2 descriptors whose distance was the smallest
            secondPairs.append([i,np.where(dist == dist.min())[0][0]])

    #We now have pairs from first and second image
    firstPairs = np.asarray(firstPairs)
    secondPairs = np.flip(secondPairs, axis=1)
    #We convert these arrays to sets, and keep only the matching pairs
    pairs = np.array([x for x in set(tuple(x) for x in firstPairs) & set(tuple(x) for x in secondPairs)])
    return pairs

#I increased the descSigma for find_matches_2 from 2 to 6 to get more descriptors
def find_matches_2(I1, I2, sigma, detector="hessian", radius=30, descSigma = 6):

    if(detector == "hessian"):
        _, fPoints1 = hessian_points(I1, sigma)
        _, fPoints2 = hessian_points(I2, sigma)
    else:
        _, fPoints1 = harris_points(I1, sigma)
        _, fPoints2 = harris_points(I2, sigma)

    #We get the descriptors from out feature points
    desc1 = a4_utils.simple_descriptors(I1, fPoints1[0], fPoints1[1], radius=radius, sigma=descSigma)
    desc2 = a4_utils.simple_descriptors(I2, fPoints2[0], fPoints2[1], radius=radius, sigma=descSigma)

    #We find symetric correspondences (desc1 matched with desc2 and the other way round)
    correspondences = find_correspondences_symetric_snd_most_similar(desc1, desc2)

    points1 = []
    points2 = []
    for x1, x2 in correspondences:
        points1.append([fPoints1[1][x1], fPoints1[0][x1]])
        points2.append([fPoints2[1][x2], fPoints2[0][x2]])

    #We return the matching points for each image
    return points1, points2


def task_c():
    I1 = np.asarray(Image.open("data\graf\graf_a_small.jpg").convert("L")).astype(np.float64) / 255
    I2 = np.asarray(Image.open("data\graf\graf_b_small.jpg").convert("L")).astype(np.float64) / 255

    points1, points2 = find_matches_2(I1, I2, 1)


    a4_utils.display_matches(I1, points1, I2, points2)

#-------------------------------------------------------------------------------------------------------------------






#TASK E-------------------------------------------------------------------------------------------------------------------
#https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
#https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

#why ORB: more efficient that SIFT and SURF, it's rotation invariant unlike FAST

def task_e():
    video = cv2.VideoCapture("data/video.mp4")
    orb = cv2.ORB_create(nfeatures=100, scoreType=cv2.ORB_FAST_SCORE)

    #Check if video opened successfully
    if (video.isOpened()== False): 
        print("Error opening the video/file")
    
    # Read until video is completed
    while(video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            key_points = orb.detect(frame, None)
            # Display the resulting frame with the keypoints on it
            cv2.imshow('video',cv2.drawKeypoints(frame, key_points, frame))
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    video.release()
    cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------------------





# task_a()
# task_b()
# task_c()
# task_e()