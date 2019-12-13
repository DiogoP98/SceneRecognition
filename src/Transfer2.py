import numpy as np
import os, fnmatch, pickle, sys
import cv2
import random

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.layers import Input, Flatten
from keras.models import Model
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle   

import warnings
warnings.filterwarnings("ignore")

def preprocessData():
    list_of_classes = []
    hash_map = {}
    index = 0
    N = 0

    max_height = 0
    max_width = 0
    for _, dirs, files in os.walk("../data/training/", topdown=False): #getting each class from data
        for name in dirs:
            list_of_classes.append(name)
            hash_map[name] = index
            index += 1
            number_files = len(fnmatch.filter(os.listdir("../data/training/" + name), '*.jpg'))

            N += number_files

    X = []
    y = []
    for current_class in list_of_classes:
        for index in range(100):
            image = load_img('../data/training/' + current_class + '/' + str(index) + '.jpg', grayscale=False, color_mode='rgb')
            image = img_to_array(image, dtype='float')
            width, height, _ = image.shape
            max_width = max(width, max_width)
            max_height = max(height, max_height)

    for current_class in list_of_classes:
        for index in range(100):
            image = load_img('../data/training/' + current_class + '/' + str(index) + '.jpg', grayscale=False, color_mode='rgb')
            image = img_to_array(image, dtype='float')
            width, height, _ = image.shape
            if width != max_width or height != max_height:
                image = cv2.resize(image, (max_width, max_height))
            
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            X.append(image)
            y.append(hash_map[current_class])
    
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, max_width, max_height

def getFeatures(X, model):
    feature_matrix = []
    for image in X:
        image_feature = model.predict(image)
        feature_matrix.append(image_feature.flatten())
    
    feature_matrix = np.asarray(feature_matrix)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    return feature_matrix

if __name__ == '__main__':
    # X, y, max_width, max_height = preprocessData()
    # #np.save("classes.npy", y)
    y = np.load("classes.npy")
    # model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(max_height, max_width, 3))
    # # #freeze layers in model
    # for layer in model.layers:
    #    layer.trainable = False
    #feature_matrix = getFeatures(X, model)

    #np.save("feature_matrix_transfer_max.npy", feature_matrix)
    feature_matrix = np.load("feature_matrix_transfer_max.npy")
    best_model = None
    best_precision = 0

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, validation_index in kf.split(feature_matrix):
        X_train, X_validation = feature_matrix[train_index], feature_matrix[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]
        
        size = X_train.shape[0]
        mlp = MLPClassifier(solver='sgd', activation='logistic', learning_rate='adaptive')
        sgd = OneVsRestClassifier(SVC(kernel='linear'))

        #number_of_iterations = 10
        #print(size)
        # for iteration in range(number_of_iterations):
        #     X_train, y_train = shuffle(X_train, y_train)
        #     previous_batch = 0
        #     for batch in range(50, size + 50, 50):
        #         mlp.partial_fit(X_train[previous_batch:batch], y_train[previous_batch:batch], np.unique(y))
        #         sgd.partial_fit(X_train[previous_batch:batch], y_train[previous_batch:batch], np.unique(y))
        #         previous_batch = batch
        #     print("Finished iteration")
        sgd.fit(X_train, y_train)
        mlp.fit(X_train, y_train)
        
        y_predicted_mlp = mlp.predict(X_validation)
        y_predicted_sgd = sgd.predict(X_validation)
        precision_sdg = precision_score(y_validation, y_predicted_sgd, average= 'micro') * 100
        precision_mlp = precision_score(y_validation, y_predicted_mlp, average= 'micro') * 100
        print("Precision SGD = " + str(precision_sdg))
        print("Precision MLP = " + str(precision_mlp))
        if precision_mlp > best_precision:
            best_precision = precision_mlp
            best_model = mlp

        if precision_sdg > best_precision:
            best_precision = precision_sdg
            best_model = sgd
    
    pickle.dump(best_model, open('Transfer_411.pickle', 'wb'))
