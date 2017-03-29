#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from class_vis import prettyPicture, output_image
from sklearn.metrics import accuracy_score
import numpy as np
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
clf = SVC(kernel='rbf', C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print("training time {}s".format(round(time()-t0, 3)))
t0 = time()
predict = clf.predict(features_test)
print("predict time {}s".format(round(time()-t0, 3)))
accuracy = accuracy_score(labels_test, predict)
print(accuracy)
print(predict[10])
print(predict[26])
print(predict[50])
ones = np.count_nonzero(predict)
zeros = len(predict) - ones
print(zeros)
print(ones)


# prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())
#########################################################
