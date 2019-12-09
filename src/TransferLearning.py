import numpy as np
import os, fnmatch, pickle
import cv2

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, Flatten
from keras.models import Model
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

def preprocessData():
    list_of_classes = []
    hash_map = {}
    index = 0
    N = 0
    max_width = 0
    max_height = 0

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
            image_resized = cv2.resize(image, (max_width, max_height))
            image_resized = np.expand_dims(image_resized, axis=0)
            image_resized = preprocess_input(image_resized)

            # if width == max_width and height == max_height:
            #     X.append(image)
            #     y.append(hash_map[current_class])
            #     continue
            
            X.append(image_resized)
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
    X, y, max_width, max_height = preprocessData()
    model = ResNet50(include_top=False, weights="imagenet", input_shape=(max_width, max_height, 3))
    #freeze layers in model
    for layer in model.layers:
        layer.trainable = False
    feature_matrix = getFeatures(X, model)
    
    #np.save("feature_matrix_inception.npy", feature_matrix)

    RBF = OneVsRestClassifier(SVC(kernel='rbf', random_state=0, C=1))
    #score = 'precision'
    #clf = GridSearchCV(RBF, parameters, scoring='%s_micro' % score)
    #mlp = MLPClassifier(activation='logistic')
    X_train, X_validation, y_train, y_validation = train_test_split(feature_matrix, y, test_size=0.3, random_state=42)
    #print(X_train.shape, X_train.shape(0))
    half_size = X_train.shape[0] // 2
    quarter_size = half_size // 2
    first_batch_X = X_train[0:quarter_size]
    first_batch_y = y_train[0:quarter_size]
    second_batch_X = X_train[quarter_size:half_size]
    second_batch_y = y_train[quarter_size:half_size]
    third_batch_X = X_train[half_size:half_size + quarter_size]
    third_batch_y = y_train[half_size:half_size + quarter_size]
    forth_batch_X = X_train[half_size + quarter_size:X_train.shape[0]]
    forth_batch_y = y_train[half_size + quarter_size:X_train.shape[0]]


    RBF.fit(first_batch_X, first_batch_y)
    RBF.fit(second_batch_X, second_batch_y)
    RBF.fit(third_batch_X, third_batch_y)
    RBF.fit(forth_batch_X, forth_batch_y)
    #mlp.fit(X_train, y_train)
    #pickle.dump(RBF, open('RBF.pickle', 'wb'))

    # print("Inception_V3:")
    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)

    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
    y_predicted = RBF.predict(X_validation)
    print(precision_score(y_validation, y_predicted, average= 'micro') * 100)
    # x = base_model.output
    # x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Dropout(0.4)(x)
    # predictions = Dense(CLASSES, activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=predictions)
