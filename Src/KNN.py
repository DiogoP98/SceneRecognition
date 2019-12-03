import os
import fnmatch
import numpy as np
import cv2

from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, average_precision_score, precision_score, accuracy_score
from sklearn.model_selection import KFold

import preprocess

def tiny_images(X, size = 16):
    tiny_images_X = []
    i = 0
    for image in X:
        width, height = image.shape
        square_size = min(width, height)
        centrex = width // 2
        centrey = height // 2
        halfcrop = square_size // 2
        croppedimage = image[centrex - halfcrop:centrex + halfcrop,centrey - halfcrop:centrey + halfcrop]
        resized_image = resize(croppedimage, (16, 16), anti_aliasing=True)
        tiny_images_X.append(resized_image.flatten())
    
    tiny_images_X = np.asarray(tiny_images_X)
    scaler = StandardScaler()
    tiny_images_X = scaler.fit_transform(tiny_images_X)

    return tiny_images_X
            
def kNearestNeighbors():
    X, y = preprocess.build_data()

    X = tiny_images(X)
    
    kf = KFold(n_splits=10, shuffle=True) 

    for n in range(1,101,1):
        averagePrecision = 0 
        for train_index, validation_index in kf.split(X):
            X_train, X_validation = X[train_index], X[validation_index]
            y_train, y_validation = y[train_index], y[validation_index]
            classifier = KNeighborsClassifier(n_neighbors=n, weights='distance')
            classifier.fit(X_train, y_train)
            y_prediction = classifier.predict(X_validation)
            averagePrecision += precision_score(y_validation, y_prediction, average= 'micro') * 100
        print("With K equal to " + str(n) + " got: " + str(averagePrecision/10))

if __name__ == '__main__':
    kNearestNeighbors()
