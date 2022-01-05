#!/usr/bin/python

import pickle
import numpy as np
np.random.seed(42)

import sys
sys.path.append("../decision_tree/")
from classifyDT import classify

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"

word_data = pickle.load( open(words_file, "r"))
word_data_cleaned = []
words_to_remove = ['sshacklensf', 'cgermannsf']
for text_from_email in word_data:
    for signature_word in words_to_remove:
        text_from_email = text_from_email.replace(signature_word, '')
    word_data_cleaned.append(text_from_email)
word_data = word_data_cleaned

authors = pickle.load( open(authors_file, "r") )

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
"""
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                            stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
"""
### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]


### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)

### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result
"""
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train = selector.transform(features_train_transformed).toarray()
features_test  = selector.transform(features_test_transformed).toarray()
"""
features_train = features_train_transformed
features_test = features_test_transformed

### your code goes here
clf = classify(features_train, labels_train)
pred = clf.predict(features_test)

accuracy = clf.score(features_train, labels_train)
print('accuracy for train data: ', accuracy)

accuracy = clf.score(features_test, labels_test)
print('accuracy for test data: ', accuracy)

## Importance of features
features_imp = clf.feature_importances_
features_imp_indices = np.array(np.where(features_imp > 0.2))[0]
features_imp2 = features_imp[features_imp_indices]

feature_names = vectorizer.get_feature_names()
#feature_names = selector.get_feature_names()
feature_names = np.array([str(x) for x in feature_names])
print 'Number of Features: {}'.format(len(feature_names))

feature_names_imp = feature_names[features_imp_indices]
print 'Important Features: {}'.format(feature_names_imp)

from collections import Counter
unique_feature_names = Counter(feature_names).keys()
unique_feature_counts = Counter(feature_names).values()


#print 'Features: \n{}'.format(feature_names)