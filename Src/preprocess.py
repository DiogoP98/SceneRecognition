import os
import fnmatch
import cv2
import numpy as np


def build_data():
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

    X = []
    y = []
    indes = 0
    for current_class in list_of_classes:
        for index in range(100):
            image = cv2.imread('../Data/training/' + current_class + '/' + str(index) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = (image / 255).astype(float)

            X.append(np.array(image))
            y.append(hash_map[current_class])
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y
