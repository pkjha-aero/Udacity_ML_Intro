#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 03:14:03 2021

@author: pkjha
"""

from __future__ import absolute_import
from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
print('stemmer.stem("responsiveness"): ', stemmer.stem("responsiveness"))
print('stemmer.stem("responsivity"): ', stemmer.stem("responsivity"))
print('stemmer.stem("unresponsiveness"): ', stemmer.stem("unresponsiveness")) 