#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 00:43:49 2021

@author: pkjha
"""

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../tools/")
sys.path.append("../naive_bayes/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



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

prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())