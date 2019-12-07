import os
import fnmatch
import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler

def get_patch(image, width, height, size, slide):
    currentx = 0
    currenty = 0

    patches = np.array([])

    while True:
        if currentx + size < width:
            limitx = currentx + size
            if currenty + size < height:
                limity = currenty + size
                patch = image[currentx : limitx, currenty : limity]
                patches = np.append(patches, patch)
            else:
                break
            currentx += slide
        else:
            currenty += slide
            currentx = 0

    return patches

def patches(size = 8, slide = 4):
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

    scaler = StandardScaler()

    for current_class in list_of_classes:
        for index in range(100):
            image = cv2.imread('../Data/training/' + current_class + '/' + str(index) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = (image / 255).astype(float)

            width, height = image.shape
            images_patches = get_patch(image, width, height, size, slide)
            images_patches = scaler.fit_transform(images_patches)
            X = np.append(X, images_patches)
            y = np.append(y, hash_map[current_class])

    return X, y, hash_map.keys()
