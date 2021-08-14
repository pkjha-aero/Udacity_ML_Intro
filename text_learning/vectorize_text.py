#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

process_data = False
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    #temp_counter = 0
    if not process_data:
        break
    print '*********************************************************************************************************'
    print 'NAME: {}, FILE POINTER: {}'.format(name, from_person)
    for path in from_person: # from_person is a file pointerfrom_sara
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if temp_counter > 0:
            path = os.path.join('..', path[:-1])
            #print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            text_from_email = parseOutText(email)
            #print 'Text from email {}: \n'.format(temp_counter), text_from_email, '\n'

            ### use str.replace() to remove any instances of the words
            for  signature_word in ["sara", "shackleton", "chris", "germani"]:
                text_from_email = text_from_email.replace(signature_word, '')
                #print 'text from email after removing signature word "',signature_word, '" : \n', text_from_email , '\n'
                
            #print 'Text from email {} after removing signature words: \n'.format(temp_counter), text_from_email , '\n\n'

            ### append the text to word_data
            word_data.append(text_from_email)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            label = 0 if name == 'sara' else 1
            from_data.append(label)
            
            email.close()
from_sara.close()
from_chris.close()

print '*********************************************************************************************************'
if process_data:
    pickle.dump( word_data, open("your_word_data.pkl", "w") )
    pickle.dump( from_data, open("your_email_authors.pkl", "w") )
    print "emails processed and written to pickle files"
else:
    word_data = pickle.load(open("your_word_data.pkl", "r") )
    from_data = pickle.load(open("your_email_authors.pkl", "r") )
    print "processed emails loaded from pickle files"
print '*********************************************************************************************************'

### in Part 4, do TfIdf vectorization here

#from nltk.corpus import stopwords
#sw = stopwords.words("english")

from sklearn.feature_extraction import text
#from sklearn.feature_extraction.text import TfidfVectorizer
my_stop_words = text.ENGLISH_STOP_WORDS.union(word_data)

vectorizer = text.TfidfVectorizer(analyzer=u'word',max_df=0.95,lowercase=True,stop_words=set(my_stop_words),max_features=150000)
#vectorizer = text.TfidfVectorizer(analyzer=u'word',max_df=0.95,lowercase=True,max_features=150000)

vectorized_data = vectorizer.fit_transform(word_data).toarray()
print 'Shape of vectorized_data: {}'.format(vectorized_data.shape)

feature_names = vectorizer.get_feature_names()
feature_names = [str(x) for x in feature_names]
print 'Number of Features: {}'.format(len(feature_names))

## FROM SKLearn Documentation
corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']
my_stop_words = text.ENGLISH_STOP_WORDS.union(corpus)

vectorizer = text.TfidfVectorizer(analyzer=u'word',stop_words=set(my_stop_words))
#vectorizer = text.TfidfVectorizer()

X = vectorizer.fit_transform(corpus).toarray()
print 'Shape of X (vectorized data): {}'.format(X.shape)

feature_names = vectorizer.get_feature_names()
feature_names = [str(x) for x in feature_names]
print 'Number of Features: {}'.format(len(feature_names))

