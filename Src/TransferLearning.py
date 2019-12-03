import numpy as np
import os, fnmatch
import cv2

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

def preprocessData():
    list_of_classes = []
    hash_map = {}
    index = 0
    N = 0
    max_width = 0
    max_height = 0

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
    # for current_class in list_of_classes:
    #     for index in range(100):
    #         image = load_img('../Data/training/' + current_class + '/' + str(index) + '.jpg', grayscale=False, color_mode='rgb', interpolation='nearest')
    #         image = img_to_array(image, dtype='float')
    #         width, height, _ = image.shape
    #         max_width = max(width, max_width)
    #         max_height = max(height, max_height)

    for current_class in list_of_classes:
        for index in range(100):
            image = load_img('../Data/training/' + current_class + '/' + str(index) + '.jpg', grayscale=False, color_mode='rgb',interpolation='nearest')
            image = img_to_array(image, dtype='float')
            image = image / 255

            # if width == max_width and height == max_height:
            #     X.append(image)
            #     y.append(hash_map[current_class])
            #     continue
            
            image_resized = cv2.resize(image, (299, 299)) 
            X.append(image_resized)
            y.append(hash_map[current_class])
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def getFeatures(X, model):
    feature_matrix = []
    index = 1
    for image in X:
        print(image.shape)
        print("here")
        image_feature = model.predict(X)
        feature_matrix.append(image_feature.flatten())
        print("Image " + str(index))
        index += 1
    
    feature_matrix = np.asarray(feature_matrix)
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    return feature_matrix

if __name__ == '__main__':
    X, y = preprocessData()
    print("Finished preprocessing data")
    model = InceptionV3(include_top=False, weights="imagenet", pooling='avg')
    feature_matrix = getFeatures(X, model)
    RBF = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
    X_train, X_validation, y_train, y_validation = train_test_split(feature_matrix, y, test_size=0.3, random_state=42)
    RBF.fit(X_train, y_train)
    y_predicted = RBF.predict(X_validation)
    print(precision_score(y_validation, y_predicted, average= 'micro') * 100)
