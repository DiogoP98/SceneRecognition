import numpy as np
import preprocess
import pickle

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
    '''

    Function which iterates over a set of images and turns them into a set of observations for K-means algorithm.
    After reading an image, it decomposes the image into an array of N patches with given patch size of (X by Y),
    it standardises these patches, and appends the patches into a set of observations X.

    :param list_of_classes: List of all classes of images.
    :param step: Pixel-wise distance between each patch's origin.
    :param patch_size: X by Y size of window (or patch) extracted from each image.
    :return: Data Matrix prepared for K-means.
    '''
    scaler = StandardScaler()
    standardised_patches = []
    images = 0
    for image in train_data:
        images += 1
        patches = view_as_windows(image, patch_size, step=step)

        for x in range(patches.shape[0]):  # Standardise all patches
            for y in range(patches.shape[1]):
                patch = scaler.fit_transform(patches[x][y])
                standardised_patches.append(patch.flatten())
    
    standardised_patches=np.asarray(standardised_patches)
    np.save('patches_matrix_8size_4step_traindata.npy', standardised_patches)

    return standardised_patches


def get_histograms(data, kmeans, patch_size, step) -> (np.ndarray):
    '''
    Receives feature matrix 'features' generated by function 'prepare for k_means', and applies K means clustering
    algorithm on each feature (image) with given K. The result of the clustering is then quantised, and turned

    :param features: Feature matrix of patches of all images.
    :param k: Number of centroids in each image.
    :return: List of all histograms (Visual Words) for each feature (image) from list of features.
    '''
    feature_vector = []
    scaler = StandardScaler()
    for image in data:
        patches = view_as_windows(image, patch_size, step=step)

        list_of_patches = []
        for x in range(patches.shape[0]):  # Standardise all patches
            for y in range(patches.shape[1]):
                patch = scaler.fit_transform(patches[x][y])
                list_of_patches.append(patch.flatten())
        
        list_of_patches = np.asarray(list_of_patches)
        predicted_clusters = kmeans.predict(list_of_patches)
        hist, bin_edges=np.histogram(predicted_clusters, bins = number_of_clusters)
        feature_vector.append(hist)
    
    feature_vector = np.asarray(feature_vector)
    return feature_vector


if __name__ == '__main__':
    # X, y = preprocess.build_data()
    # np.save("processed_data/X.npy", X)
    # np.save("processed_data/Y.npy", X)

    # kf = KFold(n_splits=5, shuffle=True)
    # number_of_clusters = 800
    # average_precision = 0
    # best_precision = 0
    # best_model = None
    # for train_index, validation_index in kf.split(X):
    #     X_train, X_validation = X[train_index], X[validation_index]
    #     y_train, y_validation = y[train_index], y[validation_index]
    #     patches = extract_patches(X_train, 8, 4)
    #     print("Finished patch extraction")

    #     kmeans = MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0).fit(patches)
    #     print("Finished K-means")

    #     train_features = get_histograms(X_train, kmeans, 8, 4)
    #     print("Finished feature vectors for training data")

    #     validation_features = get_histograms(X_validation, kmeans, 8, 4)
    #     print("Finished feature vectors for validation data")

    #     classif = OneVsRestClassifier(SVC(kernel='linear'))
    #     classif.fit(train_features, y_train)
    #     print("Finished SVM")

    #     y_predict = classif.predict(validation_features)
    #     precision = precision_score(y_validation, y_predict, average= 'micro') * 100
    #     average_precision += precision
    #     if precision > best_precision:
    #         best_model = classif
    #         best_precision = precision

    # print("Average precision: " + str(average_precision / 5))
    # pickle.dump(best_model, open('SVM_best.pickle', 'wb'))

    X, y = preprocess.build_data()
    np.save("processed_data/X.npy", X)
    np.save("processed_data/Y.npy", X)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)
    for k in range(100,1100,100):
        patches = extract_patches(X_train, 16, 8)
        print("Finished patch extraction")

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(patches)
        print("Finished K-means")

        train_features = get_histograms(X_train, kmeans, 16, 8)
        print("Finished feature vectors for training data")

        validation_features = get_histograms(X_validation, kmeans, 16, 8)
        print("Finished feature vectors for validation data")

        classif = OneVsRestClassifier(SVC(kernel='linear'))
        classif.fit(train_features, y_train)
        print("Finished SVM")

        y_predict = classif.predict(validation_features)
        precision = precision_score(y_validation, y_predict, average= 'micro') * 100
        print("With " + str(k) + " clusters: " + str(precision))



