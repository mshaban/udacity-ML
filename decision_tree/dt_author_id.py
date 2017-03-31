#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from sklearn.metrics import accuracy_score

sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from tools.prep_terrain_data import make_terrain_data
from tools.class_vis import pretty_picture
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train, labels_train, features_test, labels_test = make_terrain_data()


def dt_classifier(features_train, features_test, labels_train, labels_test):
    print(len(features_train[0]))
    # return
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time {}s".format(round(time() - t0, 3)))
    t0 = time()
    predict = clf.predict(features_test)
    print("training time {}s".format(round(time() - t0, 3)))
    accuracy = accuracy_score(predict, labels_test)
    print("Accuracy: {}".format(accuracy))
    try:
        pretty_picture(clf, features_test, labels_test)
    except ValueError:
        print("couldn't do pretty picture")


#########################################################
### your code goes here ###
dt_classifier(features_train, features_test, labels_train, labels_test)

#########################################################
