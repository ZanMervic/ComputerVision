from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from os import listdir

#Task a
def myhist3(image, n_bins):
    flat = image.reshape((-1, 3))

    #count different rgb combinations
    H = np.zeros((n_bins,n_bins,n_bins))
    for r, g, b in flat: #r / 256 * n_bins does not work because it returns a float
        x = (r * n_bins // 256)
        y = (g * n_bins // 256)
        z = (b * n_bins // 256)
        H[x, y, z] += 1

    #normalise the matrix
    return H / np.sum(H) 


#Task b
def compare_histograms(H1, H2, dist = "L"):
    result = 0
    if(dist == "L"): #Euclidean
        result = np.sqrt(np.sum((H1 - H2)**2))
    if(dist == "X"): #Chi-square
        result = (np.sum(((H1 - H2)**2)/(H1 + H2 + 1e-10))) / 2
    if(dist == "I"): #Intersection
        result = 1 - np.sum(np.minimum(H1, H2))
    if(dist == "H"): #Hellinger
        result = np.sqrt(np.sum((np.sqrt(H1) - np.sqrt(H2))**2) / 2)

    return result


def task_c():
    files = ["dataset/object_01_1.png","dataset/object_02_1.png","dataset/object_03_1.png"]
    images = []
    histograms = []

    for i, file in enumerate(files):
        images.append(np.asarray(Image.open(file)))
        histograms.append(myhist3(images[i], 8))

        plt.subplot(2,3,i + 1)
        plt.imshow(images[i])

        plt.subplot(2,3,i+4)
        plt.bar(range(8**3), histograms[i].transpose().flatten(), width=4)
        #tukej je transpose, ker njihova slika prikazuje histograme slik v bgr ne pa v rgb
        plt.title(f'L2(h1,h{i+1}) = {compare_histograms(histograms[0],histograms[i], "L"):.2f}')

    plt.show()

def dir_hist(path, n_bins):
    #Get the names of all images (files) in a directory
    images = listdir(path)
    img_data = []

    for image in images:
        I = np.asarray(Image.open(f'{path}/{image}'))
        img_data.append((myhist3(I, n_bins).flatten(), I, image))

    #Returns a tuple (histogram, image, image_name)
    return img_data

def task_d(path = "./dataset", n_bins = 8, img_path = "object_05_4.png", measure="H"):
    img_data = dir_hist(path, n_bins)
    img = [x for x in img_data if x[2] == img_path][0] #Get the image touple

    #Calculate the distance between img and all other images and combine with img_data
    img_data = [(x[0],x[1],x[2],compare_histograms(img[0], x[0], measure)) for x in img_data]

    #Sort the img_data list using the distances list
    img_data.sort(key = lambda x: x[3])

    plt.figure(figsize=(10,6))
    #Displaying the image and the 5 most similar images
    for i, (hist, I, I_name, distance) in enumerate(img_data[:6]):
        #Plotting the image
        plt.subplot(2, 6, i + 1, title = f'{I_name}')
        plt.imshow(I)
        #Plotting the histogram
        plt.subplot(2, 6, i + 1 + 6, title=f'{measure}={distance:.2f}')
        plt.bar(range(n_bins**3), hist, width=5)

    plt.show()


def task_e(path = "./dataset", n_bins = 8, image = "object_05_4.png", measure="H"):
    img_data = dir_hist(path, n_bins)
    img = [x for x in img_data if x[2] == image][0] #Get the image touple
    #Calculate the distance between img and all other images
    distances = [compare_histograms(img[0], x[0], measure) for x in img_data] #List of distances
    distances_index = sorted([(compare_histograms(img[0], x[0], measure),i) for i,x in enumerate(img_data)]) #List of tuples (distance index), used for marking lowest points
    dist_sorted = sorted(distances)

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.plot(distances)
    plt.plot([x[1] for x in distances_index[:5]],[x[0] for x in distances_index[:5]], "bo", mfc="none")

    plt.subplot(1,2,2)
    plt.plot(dist_sorted)
    plt.plot(dist_sorted[:5], "bo", mfc="none")
    plt.show()


# task_c()
# task_d()
# task_e()
