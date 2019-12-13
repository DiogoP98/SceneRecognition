import os, fnmatch, cv2
import numpy as np

hash_map = {}

def build_data():
    """Gets the images from training data and builds a matrix with it. Also hashes every class name to a number,
    which is added to the targets matrix.
    
    Returns:
        X [np.ndarray] -- Matrix with all training images
        y [np.ndarray] -- Matrix with the class of each training image
    """    
    list_of_classes = []
    index = 0
    N = 0
    for _, dirs, files in os.walk("../data/training/", topdown=False): #getting each class from data
        for name in dirs:
            list_of_classes.append(name)
            hash_map[name] = index
            index += 1
            number_files = len(fnmatch.filter(os.listdir("../Data/training/" + name), '*.jpg'))

            N += number_files

    X = []
    y = []
    for current_class in list_of_classes:
        for index in range(100):
            image = cv2.imread('../data/training/' + current_class + '/' + str(index) + '.jpg', cv2.IMREAD_GRAYSCALE)
            image = (image / 255).astype(float)

            X.append(np.array(image))
            y.append(hash_map[current_class])
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def get_test_data():
    """Gets the testing data.
    
    Returns:
        [np.ndarray] -- Matrix with all testing images
    """    
    test_data = []
    for file in sorted(os.listdir("../data/testing/"), key=lambda x: int(x.split('.')[0])):
        image = cv2.imread("../data/testing/" + file, cv2.IMREAD_GRAYSCALE)
        image = (image / 255).astype(float)

        test_data.append(np.array(image))
    
    test_data = np.asarray(test_data)

    return test_data

def generate_file(test_predictions, filename):
    """Writes the test predicitons to a file.
    
    Arguments:
        test_predictions {np.ndarray} -- Array with the predictions of each image.
        filename {string} -- Name of the file to where predictions are going to be written
    """
    f = open(filename, "w")
    image = 0
    for file in sorted(os.listdir("../data/testing/"), key=lambda x: int(x.split('.')[0])):
        predicted_class = list(hash_map.keys())[list(hash_map.values()).index(test_predictions[image])]
        f.write(file + " " + predicted_class + "\n")
        image += 1

    f.close()
