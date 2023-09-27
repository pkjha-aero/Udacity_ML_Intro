#!/usr/bin/python
import numpy as np
import pickle
#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
del data_dict['TOTAL']

features_list = ["salary", "bonus"]
data = featureFormat(data_dict, features_list)

### PLOT DATA
plt.figure(1)
plt.scatter (data[:,0], data[:,1], color = 'b')
plt.xlabel(features_list[0])
plt.ylabel(features_list[1])
#plt.show()
plt.draw()

### your code below
keys = np.array(data_dict.keys())
data_without_keys = np.array([[float(data_dict[person]["salary"]), float(data_dict[person]["bonus"])] for person in keys])

colIndex = 1
sortingIndex = data_without_keys[:,colIndex].argsort()
keys_sorted = keys[sortingIndex]
data_without_keys_sorted = data_without_keys[sortingIndex]

#len_cleaned_data = int(0.9*len(data_with_error))
#cleaned_data = data_with_error_sorted[:len_cleaned_data,:]


