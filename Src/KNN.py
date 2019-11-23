import os
import fnmatch
import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report

def tiny_images(crop = 16):
    list_of_classes = []
    hash_map = {}
    index = 0
    N = 0
    for _, dirs, files in os.walk("../Data/training/", topdown=False): #getting each class from data
        for name in dirs:
            list_of_classes.append(name)
            hash_map[name] = index
            index += 1
            number_files = len(fnmatch.filter(os.listdir("../Data/training/" + name), '*.jpg'))

            N += number_files

    X = np.array([])
    y = np.array([])

    for current_class in list_of_classes:
        for index in range(100):
            image = cv2.imread('../Data/training/' + current_class + '/' + str(index) + '.jpg', cv2.IMREAD_GRAYSCALE)
            width, height = image.shape
            centrex = (width - crop) // 2
            centrey = (height - crop) // 2
            croppedimage = image[centrex:centrex + crop,centrey:centrey + crop]
            X = np.append(X, croppedimage.flatten())
            y = np.append(y, hash_map[current_class])
    
    X = np.reshape(X, (N, crop*crop))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
            
            

def kNearestNeighbors():
    X, y = tiny_images()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=42)

    for n in range(1,1000,10):
        classifier = KNeighborsClassifier(n_neighbors=n, weights='distance')
        classifier.fit(X_train, y_train)
        #print(r2_score(y_validation, classifier.predict(X_validation)))

