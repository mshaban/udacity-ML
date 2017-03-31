#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

from time import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from tools.class_vis import pretty_picture
from tools.email_preprocess import preprocess
from tools.prep_terrain_data import make_terrain_data


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels



#########################################################
### your code goes here ###
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


def best_c(features_train, labels_train, features_test, labels_test):
    c = 1.0
    best_accuracy = 0
    cnt = 0
    features_train = features_train[:len(features_train) / 100]
    labels_train = labels_train[:len(labels_train) / 100]
    
    while True and c < 100000000:
        clf = SVC(kernel='rbf', C=c)
        clf.fit(features_train, labels_train)
        predict = clf.predict(features_test)
        new_accuracy = accuracy_score(labels_test, predict)
        print(new_accuracy)
        if new_accuracy < best_accuracy:
            cnt += 1
        if cnt >= 10:
            break
        best_accuracy = new_accuracy
        c *= 2.0
        print(c, best_accuracy)
    return c


def svc_classifier(features_train, labels_train, features_test, labels_test):
    """

    :param features_train: 
    :param labels_train: 
    :param features_test: 
    :param labels_test: 
    :return: 
    """
    clf = SVC(kernel='rbf', C=10000.0)
    t0 = time()
    # features_train = features_train[:len(features_train) / 100]
    # labels_train = labels_train[:len(labels_train) / 100]
    clf.fit(features_train, labels_train)
    print("training time {}s".format(round(time() - t0, 3)))
    t0 = time()
    predict = clf.predict(features_test)

    print("predict time {}s".format(round(time() - t0, 3)))
    accuracy = accuracy_score(labels_test, predict)
    print("Accuracy: {}".format(accuracy))

    ones = np.count_nonzero(predict)
    zeros = len(predict) - ones
    print(zeros)
    print(ones)
    # pretty_picture(clf, features_test, labels_test)


def main():
    features_train, features_test, labels_train, labels_test = preprocess()
    # features_train, labels_train, features_test, labels_test = make_terrain_data()
    svc_classifier(features_train, labels_train, features_test, labels_test)
    # c = best_c(features_train, labels_train, features_test, labels_test)
    # print(c)


if __name__ == '__main__':
    main()
