#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 22:53:12 2021

@author: pkjha
"""

#!/usr/bin/python
import sys
import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

sys.path.append("../naive_bayes/")
from studentRegression import studentReg
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



reg = studentReg(ages_train, net_worths_train)


plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())