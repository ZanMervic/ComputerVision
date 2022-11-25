from matplotlib import pyplot as plt
import cv2
from a3_utils import draw_line;
import numpy as np
import math
from PIL import Image
from exercise2 import findedges

#TASK A-------------------------------------------------------------------------------------------------------------------
def task_a(rho_bins = 300, theta_bins = 300):
    plt.figure(figsize=(8,8))

    #Array of all thetas
    thetas = np.linspace(-np.pi/2, np.pi, theta_bins)
    #Array or all our points
    points = [(10,10), (30,60), (50,20), (80,90)]
    #For each point we do this
    for i,(x, y) in enumerate(points):
        #Create a accumulator array which will be used for displaying our sinus
        acc = np.zeros((rho_bins, theta_bins))
        #Calculate our rhos for every theta (real numbers which we need to map to positive integers)
        rho = (x * np.cos(thetas) + y * np.sin(thetas))
        #We bin our rhos to the number of bins (map them to positive integers) ------------> mogoce delam narobe
        bin_rho = np.round(rho + rho_bins/2).astype(int)
        #For each theta (x axis) we increase the accumulator array on index [bin_rho(theta), theta]
        for theta in range(1,theta_bins):
            y = bin_rho[theta]
            # if(y < rho_bins and y >= 0):
            acc[y, theta] = acc[y, theta] + 1

        plt.subplot(2,2,i + 1)
        plt.imshow(acc,)

    plt.show()
#TASK A-------------------------------------------------------------------------------------------------------------------







#TASK B-------------------------------------------------------------------------------------------------------------------
def hough_find_lines(I, rho_bins, theta_bins, threshold = -1):
    if(threshold > 0):
        I = findedges(I, 1, threshold) #non-maxima-suppression
    acc = np.zeros((rho_bins, theta_bins)) #Creating a accumulator matrix
    thetas = np.linspace(-np.pi/2, np.pi/2, theta_bins) #Define theta parameter space
    D = np.sqrt(I.shape[0]**2 + I.shape[1]**2) #Define the range of rho
    #Iterate through image and for each non 0 pixel we calculate the sinusoid
    for (x, y), pixel in np.ndenumerate(I):
            if(pixel < 1):
                continue

            rho = (y * np.cos(thetas) + x * np.sin(thetas)) #Calculate the rhos (i switched x and y here because  image.shape[0] = y axis, image.shape[1] = x axis)
            #Rho + D -> moves the rho range from -rho-rho to 0-2rho
            #(rho + D) / 2D -> moves the rho range from 0 - 2rho to 0 - 1
            #((rho + D) / (2*D)) * rho_bins -> moves the rho range from 0 - 1 to 0 - nbins
            bin_rho = np.round(((rho + D) / (2*D)) * rho_bins).astype(int) 
            for theta in range(1,theta_bins):
                y = bin_rho[theta]
                # if(y < rho_bins and y >= 0):
                acc[y, theta] = acc[y, theta] + 1 #Increase the accumulator matrix cells
    return acc

def task_b():
    plt.figure(figsize=(10, 6))
    images = ["synthetic.png", "oneline.png", "rectangle.png"]
    for i,image in enumerate(images):
        I = np.asarray(Image.open(f"./images/{image}").convert("L")).astype(np.float64)
        plt.subplot(1,3,i+1)
        plt.imshow(hough_find_lines(I, 300, 300, 20))
        plt.title(image)

    plt.show()
#-------------------------------------------------------------------------------------------------------------------







#TASK C-------------------------------------------------------------------------------------------------------------------

#We go through te accumulator array we get from hough and try to keep only
#The points with the most votes (the actual edges)
def nonmaxima_suppression_box(I):
    #The idea is similar to the 2d median from assignment2
    result = np.zeros_like(I) #Create a 0s array 
    I = np.pad(I, 1) #Pad I with 1 layer of 0s
    for j in range(result.shape[0]):
        for i in range(result.shape[1]):
            if(I[j+1,i+1] == 0):
                continue
            #If the pixel is higher than 0 we compare it to the pixels around it
            temp = np.copy(I[j:j+3,i:i+3])
            temp[1,1] = 0
            max_neighbor = np.max(temp)
            if(I[j+1,i+1] > max_neighbor):
                #If the pixel is higher than all of it's neighbors we keep it
                result[j,i] = I[j+1,i+1]
            elif(I[j+1,i+1] == max_neighbor):
                I[j+1,i+1] = 0

    return result

def task_c():
    images = ["synthetic.png","oneline.png", "rectangle.png"]
    plt.figure(figsize=(10, 6))
    for i,image in enumerate(images):
        I = np.asarray(Image.open(f"./images/{image}").convert("L")).astype(np.float64)
        I_lines = hough_find_lines(I, 300, 300, 20)

        plt.subplot(2,3,i+1)
        plt.imshow(I_lines)
        plt.title(image)

        plt.subplot(2,3,i+4)
        plt.imshow(nonmaxima_suppression_box(I_lines))
        plt.title(f"lines - {image}")

    plt.show()
#-------------------------------------------------------------------------------------------------------------------






#TASK D-------------------------------------------------------------------------------------------------------------------

#I = normal image (just for size), I_lines = acc matrix (preferably nonmax-suppression)
def rho_theta_pairs(I, I_acc_max, theta_bins, rho_bins, threshold = 500):
    pairs = [] #Here we will sotre (rho,theta) pairs
    thetas = np.linspace(-np.pi/2, np.pi/2, theta_bins) #Define the theta ranges (needed to convert from binned thata to actual value)
    D = np.sqrt(I.shape[0]**2 + I.shape[1]**2) #Get the image diagonal (for converting binned rho to float)
    #Extract all theta and rho pairs (cords) that are left after nonmaxima and higher than the threshold
    for cords, pixel in np.ndenumerate(I_acc_max):
        if(pixel < threshold):
            continue
        #Reverse the theta and rho binning
        theta = thetas[cords[1]]
        rho = ((cords[0] * 2 * D)/rho_bins) - D
        pairs.append((rho,theta)) 

    return pairs


def task_d():
    images = ["synthetic.png","oneline.png", "rectangle.png"]
    plt.figure(figsize=(10, 6))
    theta_bins = 300 #Define the number of theta bins
    rho_bins = 300 #Define the number of rho bins
    for i,image in enumerate(images):
        I = np.asarray(Image.open(f"./images/{image}").convert("L")).astype(np.float64)
        
        #Get the accumulator matrix with hough algorithm
        if(i == 0):
            #For the synthetic image the hough_find_lines must be done without the findedges function
            #Or else it will create a bad image
            acc = hough_find_lines(I, theta_bins, rho_bins)
        else:
            acc = hough_find_lines(I, theta_bins, rho_bins, 20) 
        
        #Do the nonmaxima_suppression on the accumulator matrix
        I_acc_max = nonmaxima_suppression_box(acc) 
        
        threshold = 2 #Threshold for synthetic
        if(i == 1):
            threshold = 1000 #Threshold for oneline
        if(i == 2):
            threshold = 342 #Threshold for rectangle
        pairs = rho_theta_pairs(I, I_acc_max, theta_bins, rho_bins, threshold=threshold) 

        plt.subplot(1,3,i+1)
        plt.imshow(I, cmap="gray")
        for pair in pairs:
            draw_line(pair[0], pair[1], I.shape[0], I.shape[1])
        plt.title(image)

    plt.show()
#-------------------------------------------------------------------------------------------------------------------







#TASK E-------------------------------------------------------------------------------------------------------------------
def task_e(theta_bins = 300, rho_bins = 300):
    images = ["bricks.jpg", "pier.jpg"]
    plt.figure(figsize=(10, 6))
    for i, image in enumerate(images):
        I_color = cv2.cvtColor(cv2.imread(f'images/{image}'), cv2.COLOR_BGR2RGB)
        I_gray = cv2.cvtColor(I_color, cv2.COLOR_RGB2GRAY)
        I_edges = cv2.Canny(I_gray, 150, 250)
        I_acc = hough_find_lines(I_edges, theta_bins, rho_bins)
        I_acc_max = nonmaxima_suppression_box(I_acc)

        pairs = rho_theta_pairs(I_gray, I_acc_max, theta_bins, rho_bins, 950 - (400 * i))

        plt.subplot(2,5,(i*5)+1)
        plt.imshow(I_color)
        plt.title(image)

        plt.subplot(2,5,(i*5)+2)
        plt.imshow(I_edges)
        plt.title(image)

        plt.subplot(2,5,(i*5)+3)
        plt.imshow(I_acc)
        plt.title(image)

        plt.subplot(2,5,(i*5)+4)
        plt.imshow(I_acc_max)
        plt.title(image)

        plt.subplot(2,5,(i*5)+5)
        plt.imshow(I_color)
        for pair in pairs:
            draw_line(pair[0], pair[1], I_color.shape[0], I_color.shape[1])

    plt.show()
#-------------------------------------------------------------------------------------------------------------------







#Task H ------------------------------------------------------------------------------------------------------------
#If we want to calculate the max line length, we need to calculate where the line
# x*cos(theta) + y*sin(theta) = rho would hit the border lines x = 0, x = width, y = 0 and y = height

# x = 0 -> y*sin(theta) = rho -> y = rho/sin(theta)
# x = width -> width*cos(theta) +y*sin(theta) = rho -> y = (rho - width*cos(theta)) / sin(theta)
# y = 0 -> x*cos(theta) = rho -> x = rho/cos(theta)
# y = height -> x*cos(theta) +height*sin(theta) = rho -> x = (rho - height*sin(theta)) / cos(theta)

#Only one of the x and one of the y will be in range of our image size
#Those are the coordinates we will use to calculate our max line length
def norm(rho, theta, width, heigth):

    valid_intersections = []
    
    y1 = rho/np.sin(theta)
    if 0 <= y1 and y1 < heigth: 
        valid_intersections.append((0, y1))
    y2 = (rho - width*np.cos(theta))/np.sin(theta)
    if 0 <= y2 and y2 < heigth: 
        valid_intersections.append((width, y2))
    x1 = rho/np.cos(theta)
    if 0 <= x1 and x1 < width: 
        valid_intersections.append((x1, 0))
    x2 = (rho - heigth*np.sin(theta))/np.cos(theta)
    if 0 <= x2 and x2 < width: 
        valid_intersections.append((x2, heigth))
    
    #Calculate the Euclidean distance
    length = np.sqrt((valid_intersections[0][0] - valid_intersections[1][0])**2 +
     (valid_intersections[0][1] - valid_intersections[1][1])**2)
    if length == 0: return 0
    
    return 1/length


def hough_find_lines_norm(I, rho_bins, theta_bins):
    acc = np.zeros((rho_bins, theta_bins)) #Creating a accumulator matrix
    thetas = np.linspace(-np.pi/2, np.pi/2, theta_bins) #Define theta parameter space
    D = np.sqrt(I.shape[0]**2 + I.shape[1]**2) #Define the range of rho
    #Iterate through image and for each non 0 pixel we calculate the sinusoid
    for (x, y), pixel in np.ndenumerate(I):
            if(pixel < 1):
                continue
            #Calculate the rhos (i switched x and y here because  image.shape[0] = y axis, image.shape[1] = x axis)
            rhos = (y * np.cos(thetas) + x * np.sin(thetas)) 
            #Rho + D -> moves the rho range from -rho-rho to 0-2rho
            #(rho + D) / 2D -> moves the rho range from 0 - 2rho to 0 - 1
            #((rho + D) / (2*D)) * rho_bins -> moves the rho range from 0 - 1 to 0 - nbins
            bin_rhos = np.round(((rhos + D) / (2*D)) * rho_bins).astype(int) 
            for bin_theta in range(1,theta_bins):
                #Get the real rho and theta
                rho = rhos[bin_theta] #real rho value
                theta = thetas[bin_theta] #real theta value
                bin_rho = bin_rhos[bin_theta] #binned rho value
                # if(bin_rho < rho_bins and bin_rho >= 0):
                #Increase the accumulator matrix cells
                acc[bin_rho, bin_theta] = acc[bin_rho, bin_theta] + norm(rho, theta, I.shape[1], I.shape[0]) 
    return acc


def task_h(theta_bins = 300, rho_bins = 300, image="pier.jpg"):
    plt.figure(figsize=(10, 6))
    for i in range(0,2):
        I_color = cv2.cvtColor(cv2.imread(f'images/{image}'), cv2.COLOR_BGR2RGB)
        I_gray = cv2.cvtColor(I_color, cv2.COLOR_RGB2GRAY)
        I_edges = cv2.Canny(I_gray, 150, 250)
        if(i == 0):
            I_acc = hough_find_lines(I_edges, theta_bins, rho_bins)
        if(i == 1):
            I_acc = hough_find_lines_norm(I_edges, theta_bins, rho_bins)
        I_acc_max = nonmaxima_suppression_box(I_acc)

        if(i == 0):
            pairs = rho_theta_pairs(I_gray, I_acc_max, theta_bins, rho_bins, 550)
        if(i == 1):
            pairs = rho_theta_pairs(I_gray, I_acc_max, theta_bins, rho_bins, 1.05)

        plt.subplot(2,3,(i*3)+1)
        plt.imshow(I_acc)
        plt.title(image)

        plt.subplot(2,3,(i*3)+2)
        plt.imshow(I_acc_max)
        plt.title(image)

        plt.subplot(2,3,(i*3)+3)
        plt.imshow(I_color)
        for pair in pairs:
            draw_line(pair[0], pair[1], I_color.shape[0], I_color.shape[1])

    plt.show()



# task_a()
# task_b()
# task_c()
# task_d()
# task_e()
task_h()