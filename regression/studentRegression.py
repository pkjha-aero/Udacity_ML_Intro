#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 22:54:00 2021

@author: pkjha
"""

def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    from sklearn.linear_model import LinearRegression
    ### name your regression reg
    reg = LinearRegression()
    ### your code goes here!
    
    reg.fit (ages_train, net_worths_train)
    
    
    return reg