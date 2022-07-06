import numpy as np
import cv2
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

def allCategories():
    return ['rock', 'Palm', 'Fist', 'swing', 'L', 'okay', 'unknown']

# input path of image and return ready image
# read image then normalize values [0,255] to [0,1]
# then resize to 100,120 then add (100,120,1)
def procImage(path): 
    image = cv2.imread(path,0)
    image = image/255.
    image = cv2.resize(image,(100,120))
    image = image.reshape(image.shape[0],image.shape[1],1)
    return image

# Input category and return one hot array of it 
# Exmpl input: 'Palm' return [0,1,0,0,0,0]
def oneHot(category):
    CATEGORIES = allCategories()
    size = len(CATEGORIES)
    result = np.zeros(size, dtype=int)
    for i in range(size):
        if category == CATEGORIES[i]:
            result[i] = 1
    return result

# Input one hot array and return category of it 
# Exmpl input: [0,0,0,0,1,0] return 'L'
def getCategory(list_val):
    CATEGORIES = allCategories()
    x= np.argmax(list_val)
    return CATEGORIES[x]

# Input directory of images
# Use the functions maintioned in before 'oneHot' and 'procImage'
# to generate datasets X,y
def getData(path):
    CATEGORIES = allCategories()
    Xs = []
    Ys = []
    for category in CATEGORIES:
        folder = os.path.join(path,category)
        for file in tqdm(os.listdir(folder)):
            fullPath = os.path.join(folder,file)
            Xs.append(procImage(fullPath))
            y = oneHot(category)
            Ys.append(y)
    return  np.array(Xs), np.array(Ys)

# Show n images with labes
# Input arrays X of images and y and number of images to show
def show(X,Y,size):
    plt.figure(figsize=(20,20))
    for i in range(size):
        plt.subplot(5,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i,:, :, 0], cmap='gray')
        plt.xlabel(getCategory(Y[i]))
    plt.show()
    
# Sava pickle file out of data 
def save(data,path):
    pickle_out = open(path,"wb")
    pickle.dump(data,pickle_out)
    pickle_out.close()