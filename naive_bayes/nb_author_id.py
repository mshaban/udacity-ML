#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from time import time

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

from tools.class_vis import pretty_picture
from tools.email_preprocess import preprocess
from tools.prep_terrain_data import make_terrain_data


def nb_classifier(features_train, labels_train, features_test, labels_test):
    """
    
    :param features_train: 
    :param labels_train: 
    :param features_test: 
    :param labels_test: 
    :return: 
    """
    
    clf = GaussianNB()
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time {}s".format(round(time() - t0, 3)))
    t0 = time()
    predict = clf.predict(features_test)

    print("predict time {}s".format(round(time() - t0, 3)))
    accuracy = accuracy_score(labels_test, predict)
    print("Accuracy: {}".format(accuracy))
    pretty_picture(clf, features_test, labels_test)


#########################################################


def main():
    features_train, features_test, labels_train, labels_test = preprocess()
    x_train, y_train, x_test, y_test = make_terrain_data()
    nb_classifier(x_train, y_train, x_test, y_test)
    # nb_classifier(features_train, labels_train, features_test, labels_test)


if __name__ == '__main__':
    main()
