import os
import fnmatch
import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, average_precision_score, precision_score, accuracy_score

import preprocess

def tiny_images(X, crop = 16):
    tiny_images_X = np.array([])

    for image in X:
        width, height = image.shape
        centrex = width // 2
        centrey = height // 2
        halfcrop = crop // 2
        croppedimage = image[centrex - halfcrop:centrex + halfcrop,centrey - halfcrop:centrey + halfcrop]
        tiny_images_X = np.append(tiny_images_X, croppedimage.flatten())
    
    tiny_images_X = np.reshape(tiny_images_X, (X.shape[0], crop*crop))
    scaler = StandardScaler()
    tiny_images_X = scaler.fit_transform(tiny_images_X)

    return tiny_images_X
            
def kNearestNeighbors():
    X, y = preprocess.build_data()

    X = tiny_images(X)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

    best = 0
    best_k = 0
    for n in range(1,200,1):
        classifier = KNeighborsClassifier(n_neighbors=n, weights='distance')
        classifier.fit(X_train, y_train)
        y_prediction = classifier.predict(X_validation)
        #print(classification_report(y_validation, classifier.predict(X_validation), target_names= classes))
        if precision_score(y_validation, y_prediction, average= 'micro') * 100 > best:
            best_k = n
            best = precision_score(y_validation, y_prediction, average= 'micro') * 100
        
    print("Best presicion: " + str(best) + ", with :" + str(best_k))

if __name__ == '__main__':
    kNearestNeighbors()
