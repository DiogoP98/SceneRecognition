from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier

trainImgs, trainLabels = make_classification(n_features=4, n_classes=15,random_state=0)

def SVMlinear(trainImgs, trainLabels, testImgs):
    # Need to do hyperparameter tuning for the value of C (penalty parameter for the error term) 
    # Value may range between 1.0-5000.0
    clf = OneVsRestClassifier(LinearSVC(C=1.0, multi_class='ovr', max_iter=1000))
    clf.fit(trainImgs, trainLabels)
    testLabels = clf.predict(testImgs)
    # List of predicted labels for the test images
    return testLabels







