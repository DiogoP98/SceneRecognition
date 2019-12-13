import numpy as np
import os, fnmatch, pickle, sys
import cv2
import random

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle 

import preprocess
import warnings
warnings.filterwarnings("ignore")

hash_map = {}
max_height = 0
max_width = 0

def preprocessData():
    """Get training images and resize each one of them to get the size of the bigger image in the data set
    
    Returns:
        X [np.ndarray] -- Matrix with all the images, resized.
        y [type] -- Matrix 1*N, with the class of each image.
    """    
    list_of_classes = []
    index = 0

    #getting each class from data
    for _, dirs, files in os.walk("../data/training/", topdown=False): 
        for name in dirs:
            list_of_classes.append(name)
            hash_map[name] = index
            index += 1
            number_files = len(fnmatch.filter(os.listdir("../data/training/" + name), '*.jpg'))

    X = []
    y = []
    #get the size of the biggest image
    for current_class in list_of_classes:
        for index in range(100):
            image = load_img('../data/training/' + current_class + '/' + str(index) + '.jpg', grayscale=False, color_mode='rgb')
            image = img_to_array(image, dtype='float')
            width, height, _ = image.shape
            max_width = max(width, max_width)
            max_height = max(height, max_height)

    #resizing every image
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

    return X, y

def get_and_resize_test_data(width, height):
    """Gets the test data and resize it.
    
    Arguments:
        width {int} -- Width of the resized image
        height {int} -- Height of the resized image
    
    Returns:
        [np.ndarray] -- Matrix with all the images, resized.
    """    
    test_data = []
    for file in sorted(os.listdir("../data/testing/"), key=lambda x: int(x.split('.')[0])):
        image = load_img("../data/testing/" + file, grayscale=False, color_mode='rgb')
        image = img_to_array(image, dtype='float')

        width, height, _ = image.shape
        if width != max_width or height != max_height:
            image = cv2.resize(image, (max_width, max_height))
            
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        test_data.append(np.array(image))
    
    test_data = np.asarray(test_data)

    return test_data


def getFeatures(X, model):
    """Gets the feature of each image and ataches it to an array, after flattening it.
    
    Arguments:
        X {np.ndarray} -- Matrix with all the images
        model {keras.applications.resnet_v2.ResNet50V2} -- The pre-trained model that it's going to feature extract each image
    
    Returns:
        [np.ndarray] -- Matrix with the features extracted of each image in each row.
    """    
    feature_matrix = []
    for image in X:
        image_feature = model.predict(image)
        feature_matrix.append(image_feature.flatten())
    
    feature_matrix = np.asarray(feature_matrix)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    return feature_matrix

def generate_file(test_predictions):
    """Writes the test predicitons to a file.
    
    Arguments:
        test_predictions {np.ndarray} -- Array with the predictions of each image.
        hash_map {dictionary} -- Correspondence between the class number and class name
    """    
    f = open("run3.txt", "w")
    image = 0
    for file in sorted(os.listdir("../data/testing/"), key=lambda x: int(x.split('.')[0])):
        predicted_class = list(hash_map.keys())[list(hash_map.values()).index(test_predictions[image])]
        f.write(file + " " + predicted_class + "\n")
        image += 1

    f.close()

if __name__ == '__main__':
    X, y = preprocessData()
    np.save("classes.npy", y)

    model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(max_width, max_height, 3))
    #freeze layers in model
    for layer in model.layers:
        layer.trainable = False

    feature_matrix = getFeatures(X, model)
    np.save("feature_matrix_transfer_411.npy", feature_matrix)

    best_model = None
    best_precision = 0

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, validation_index in kf.split(feature_matrix):
        X_train, X_validation = feature_matrix[train_index], feature_matrix[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]
        
        size = X_train.shape[0]
        mlp = MLPClassifier(solver='sgd', activation='logistic', learning_rate='adaptive')
        sgd = SGDClassifier()

        #Partial fit the batches in 15 iterations
        number_of_iterations = 15
        for iteration in range(number_of_iterations):
            X_train, y_train = shuffle(X_train, y_train)
            previous_batch = 0
            for batch in range(50, size + 50, 50):
                mlp.partial_fit(X_train[previous_batch:batch], y_train[previous_batch:batch], np.unique(y))
                sgd.partial_fit(X_train[previous_batch:batch], y_train[previous_batch:batch], np.unique(y))
                previous_batch = batch
        
        y_predicted_mlp = mlp.predict(X_validation)
        y_predicted_sgd = sgd.predict(X_validation)
        precision_sdg = precision_score(y_validation, y_predicted_sgd, average= 'micro') * 100
        precision_mlp = precision_score(y_validation, y_predicted_mlp, average= 'micro') * 100

        if precision_mlp > best_precision:
            best_precision = precision_mlp
            best_model = mlp

        if precision_sdg > best_precision:
            best_precision = precision_sdg
            best_model = sgd
    
    pickle.dump(best_model, open('Transfer_411.pickle', 'wb'))
    #Start Test prediction
    test_data = get_and_resize_test_data(max_width, max_height) 
    print("Finished image data")
    features_test = getFeatures(test_data, model)
    print("Finish feature extraction")
    test_predictions = best_model.predict(features_test)
    print("Finish predictions")
    generate_file(test_predictions, hash_map)

