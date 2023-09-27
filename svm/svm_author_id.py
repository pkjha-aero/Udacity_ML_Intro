#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
sys.path.append("../naive_bayes/")
from email_preprocess import preprocess
from class_vis import prettyPicture


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Slice training data to just 1%
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

#########################################################

#########################################################
from sklearn.svm import SVC

### create classifier
#clf = SVC(kernel="linear")
#clf = SVC(kernel="linear", gamma = 1.0, C = 1.0)
clf = SVC(kernel="rbf", C = 10000.0)
#clf = SVC(kernel="rbf", gamma = 1000.0, C = 1.0)
#clf = SVC(kernel="poly", gamma = 1.0)

#########################################################

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

### calculate and return the accuracy on the test data
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print('Accuracy Score: {}'.format(acc))

#prettyPicture(clf, features_test[:,0], labels_test)