#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 00:43:43 2021

@author: pkjha
"""

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion = 'gini', random_state=0, min_samples_split=40)
    clf.fit(features_train, labels_train)
    
    return clf