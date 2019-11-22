import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

def tiny_images():
    list_of_classes = []
    for _, dirs, _ in os.walk("../Data", topdown=False): #getting each class from data
        for name in dirs:
            list_of_classes.append(name)
    
    X = np.array()
    y = np.array()
    for current_class in list_of_classes:
        for index in range(100):
            image = cv2.imread('../Data/' + current_class + '/' + str(index) + '.jpg', cv2.IMREAD_UNCHANGED)
            width, height = image.shape
            centrex = width // 2
            centrey = height // 2 
            croppedimage = image[centrex:-centrex,centrey:centrey]
            
            X.append(croppedimage.flatten(), axis = 0)
            y.append(current_class, axis = 0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
            
            

def kmeans():
    print("ok")
