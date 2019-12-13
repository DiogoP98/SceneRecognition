import os, cv2
import numpy as np

from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score

import auxiliar

def tiny_images(X, size = 16):
    """Centre-crops and resizes an image.
    
    Arguments:
        X {np.ndarray} -- Dataset
    
    Keyword Arguments:
        size {int} -- Size of the resized Image (default: {16})
    
    Returns:
        [np.ndarray] -- Processed data, where each row is a flatten image.
    """    
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
    """Implements K-nearest neighbours
    """    
    X, y = auxiliar.build_data()

    X = tiny_images(X)
    
    kf = KFold(n_splits=10, shuffle=True) 

    best_precision = 0
    best_k = 0
    for n in range(1,101,1):
        averagePrecision = 0

        #Ten-fold cross validation
        for train_index, validation_index in kf.split(X):
            X_train, X_validation = X[train_index], X[validation_index]
            y_train, y_validation = y[train_index], y[validation_index]

            classifier = KNeighborsClassifier(n_neighbors=n, weights='distance')
            classifier.fit(X_train, y_train)

            y_prediction = classifier.predict(X_validation)
            precision = precision_score(y_validation, y_prediction, average= 'micro') * 100
            averagePrecision += precision
            if precision > best_precision:
                best_k = n
                best_precision = precision
                best_model = classifier
        print("With K equal to " + str(n) + " got: " + str(averagePrecision/10))
    
    print("Best model was with n = " + str(best_k) + " = " + str(best_precision))

    test_data = auxiliar.get_test_data()
    test_tiny = tiny_images(test_data)
    test_predictions = best_model.predict(test_tiny)
    auxiliar.generate_file(test_predictions, "run1.txt")

if __name__ == '__main__':
    kNearestNeighbors()
