#!/usr/bin/python
import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner

### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit (ages_train, net_worths_train)

print "score of train data: ", reg.score(ages_train, net_worths_train)
print "score of test data : ", reg.score(ages_test, net_worths_test)
print "Slope of regression: ", reg.coef_
print "Intercept of regression: ", reg.intercept_

### PLOTS
train_color = "b"
test_color = "r"
plt.scatter(ages_train, net_worths_train, color=train_color, label="train")
plt.scatter(ages_test, net_worths_test, color=test_color, label="test")
try:
    #plt.plot(ages_test, reg.predict(ages_test), color="black", label="Reg Fit")
    plt.plot(ages, reg.predict(ages), color="black", label="Reg Fit")
except NameError:
    pass
plt.xlabel("Ages")
plt.ylabel("Net Worth")
plt.legend()
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"

### RE-TRAIN WITH CLEANED DATA
### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages_train, net_worths_train, errors = zip(*cleaned_data)
    ages_train       = numpy.reshape( numpy.array(ages_train), (len(ages_train), 1))
    net_worths_train = numpy.reshape( numpy.array(net_worths_train), (len(net_worths_train), 1))

    print "Dealing with cleaned data now... ..."
    plt.scatter(ages_train, net_worths_train, color=train_color, label="train")
    plt.scatter(ages_test, net_worths_test, color=test_color, label="test")
    ### refit your cleaned data!
    try:
        reg.fit(ages_train, net_worths_train)
        plt.plot(ages_train, reg.predict(ages_train), color="black", label="Reg w/ Clean Data")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
     
    plt.xlabel("Ages")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.show()
    
    print "score of train data: ", reg.score(ages_train, net_worths_train)
    print "score of test data : ", reg.score(ages_test, net_worths_test)
    print "Slope of regression: ", reg.coef_
    print "Intercept of regression: ", reg.intercept_

else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"

