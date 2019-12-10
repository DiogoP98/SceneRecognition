import numpy as np
import os, fnmatch, pickle, sys
import cv2

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
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

classes = []
def preprocessData():
    list_of_classes = []
    hash_map = {}
    index = 0
    N = 0
    min_width = sys.maxsize
    min_height = sys.maxsize

    for _, dirs, files in os.walk("../data/training/", topdown=False): #getting each class from data
        for name in dirs:
            list_of_classes.append(name)
            hash_map[name] = index
            classes.append(index)
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
            max_width = min(width, max_width)
            max_height = min(height, max_height)

    for current_class in list_of_classes:
        for index in range(100):
            image = load_img('../data/training/' + current_class + '/' + str(index) + '.jpg', grayscale=False, color_mode='rgb')
            image = img_to_array(image, dtype='float')
            width, height, _ = image.shape
            if width != min_width or height != min_height:
                square_size = min(image.shape[0], image.shape[1])
                centrex = image.shape[0] // 2
                centrey = image.shape[1] // 2
                halfcrop = square_size // 2
                image = image[centrex - halfcrop:centrex + halfcrop,centrey - halfcrop:centrey + halfcrop]
                image = cv2.resize(image, (min_width, min_height))
            
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            X.append(image)
            y.append(hash_map[current_class])
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y, min_width, min_height

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
    X, y, min_width, min_height = preprocessData()
    #np.save("classes.npy", y)
    model = ResNet50(include_top=False, weights="imagenet", input_shape=(min_height, min_width, 3))
    #freeze layers in model
    for layer in model.layers:
       layer.trainable = False
    feature_matrix = getFeatures(X, model)
    
    #np.save("feature_matrix_transfer_min.npy", feature_matrix)
    #print("Finished Feature extraction")
    #feature_matrix = np.load("feature_matrix_transfer_min.npy")

    #y = np.load("classes.npy")

    #RBF = OneVsRestClassifier(SVC(kernel='rbf', random_state=0, C=1))
    #score = 'precision'
    #clf = GridSearchCV(RBF, parameters, scoring='%s_micro' % score)
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, validation_index in kf.split(feature_matrix):
        X_train, X_validation = feature_matrix[train_index], feature_matrix[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]
        parameter_space = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
        mlp = MLPClassifier()
        clf = GridSearchCV(mlp, parameter_space, scoring='precision_micro')
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_validation)
        print("Best parameters for MLP: ")
        print(clf.best_params_)
        print("Best score: " + str(clf.best_score_))

        parameter_space_2 = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
        svc = SVC()
        clf2 = GridSearchCV(svc, parameter_space_2, scoring='precision_micro')
        clf2.fit(X_train, y_train)
        y_predicted = clf2.predict(X_validation)
        print("Best parameters for SVM: ")
        print(clf2.best_params_)
        print("Best score: " + str(clf2.best_score_))
