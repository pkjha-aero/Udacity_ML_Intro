#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
sys.path.append("../naive_bayes/")

from class_vis import prettyPicture, output_image
from email_preprocess import preprocess
from classifyDT import classify


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)

pred = clf.predict(features_test)

### calculate and return the accuracy on the test data
#from sklearn.metrics import accuracy_score
#acc = accuracy_score(pred, labels_test)

accuracy = clf.score(features_test, labels_test)
print('accuracy: ', accuracy)

#### grader code, do not modify below this line

#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())

#########################################################


