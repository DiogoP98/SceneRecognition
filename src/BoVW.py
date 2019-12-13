import numpy as np
import auxiliar
import pickle, os

from skimage.util.shape import view_as_windows
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold


def extract_patches(train_data, patch_size, step) -> (np.ndarray):
    """Gets all the patches from training data, for further use in K-means.
    
    Arguments:
        train_data {np.ndarray} -- images in the training data.
        patch_size {int} -- size of the patches.
        step {int} -- distance between different patches.
    
    Returns:
        [np.ndarray] -- matrix, where each row is a flatten list of all patches of an image.
    """    
    scaler = StandardScaler()
    standardised_patches = []
    images = 0
    for image in train_data:
        images += 1
        patches = view_as_windows(image, patch_size, step=step)

        #Standardize the patches. All patches are going to have 0 mean and unit variance
        for x in range(patches.shape[0]): 
            for y in range(patches.shape[1]):
                patch = scaler.fit_transform(patches[x][y])
                standardised_patches.append(patch.flatten())
    
    standardised_patches=np.asarray(standardised_patches)
    #np.save('patches_matrix_8size_4step_traindata.npy', standardised_patches)

    return standardised_patches


def get_histograms(data, kmeans, patch_size, step, number_of_clusters) -> (np.ndarray):
    """Builds a matrix composed by the histograms of each image.
    
    Arguments:
        data {np.ndarray} -- matrix with all the data.
        kmeans {sklearn.cluster.KMeans} -- codebook, generated previously.
        patch_size {int} -- size of the patches.
        step {int} -- distance between different patches.
        number_of_clusters {int} -- number of clusters in k-means.
    
    Returns:
        [np.ndarray] -- matrix with the histogram of each image.
    """    
    feature_vector = []
    scaler = StandardScaler()
    for image in data:
        patches = view_as_windows(image, patch_size, step=step)

        list_of_patches = []
        for x in range(patches.shape[0]):
            for y in range(patches.shape[1]):
                patch = scaler.fit_transform(patches[x][y])
                list_of_patches.append(patch.flatten())
        
        list_of_patches = np.asarray(list_of_patches)
        predicted_clusters = kmeans.predict(list_of_patches)
        hist, bin_edges=np.histogram(predicted_clusters, bins = number_of_clusters)
        feature_vector.append(hist)
    
    feature_vector = np.asarray(feature_vector)
    return feature_vector

def generate_file(test_predictions, hash_map):
    """Writes the test predicitons to a file.
    
    Arguments:
        test_predictions {np.ndarray} -- Array with the predictions of each image.
        hash_map {dictionary} -- Correspondence between the class number and class name
    """    
    f = open("run2.txt", "w")
    image = 0
    for file in sorted(os.listdir("../data/testing/"), key=lambda x: int(x.split('.')[0])):
        predicted_class = list(hash_map.keys())[list(hash_map.values()).index(test_predictions[image])]
        f.write(file + " " + predicted_class + "\n")
        image += 1

    f.close()


if __name__ == '__main__':
    X, y = auxiliar.build_data()

    number_of_clusters = 600
    precision = 0
    best_precision = 0
    best_model = None
    best_kmeans = None
    
    #3 fold cross validation
    kf = KFold(n_splits=3, shuffle=True)
    for train_index, validation_index in kf.split(X):
        X_train, X_validation = X[train_index], X[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]
        patches = extract_patches(X_train, 4, 2)
        print("Finished patch extraction")

        #Builds codebook
        kmeans = MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0).fit(patches)
        print("Finished K-means")

        train_features = get_histograms(X_train, kmeans, 4, 2, number_of_clusters)
        print("Finished feature vectors for training data")

        validation_features = get_histograms(X_validation, kmeans, 4, 2, number_of_clusters)
        print("Finished feature vectors for validation data")

        classif = OneVsRestClassifier(SVC(kernel='linear'))
        classif.fit(train_features, y_train)
        print("Finished SVM")

        y_predict = classif.predict(validation_features)
        precision += precision_score(y_validation, y_predict, average= 'micro') * 100

    print("Average precision: " + str(precision/3))
    
    test_data = auxiliar.get_test_data()
    test_features = get_histograms(test_data, kmeans, 4, 2, number_of_clusters)
    test_predictions = classif.predict(test_features)
    auxiliar.generate_file(test_predictions, "run2.txt")



