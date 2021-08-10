#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
#features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

finance_features = np.array(finance_features)
#Scale features
finance_features[:, 0] = finance_features[:, 0]/ (1111258.0 - 477.0)
finance_features[:, 1] = finance_features[:, 1]/ (34348384.0 - 3285.0)

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
plt.figure()
for f1, f2 in finance_features:
    plt.scatter( f1, f2)
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.show()

"""
plt.figure()
for f1, f2, f3 in finance_features:
    plt.scatter( f1, f3)
plt.xlabel(feature_1)
plt.ylabel(feature_3)
plt.show()

plt.figure()
for f1, f2, f3 in finance_features:
    plt.scatter( f2, f3)
plt.xlabel(feature_2)
plt.ylabel(feature_3)
plt.show()
"""
### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(finance_features)
print "k-means labels: ", kmeans.labels_
print "k-means cluster centers: ", kmeans.cluster_centers_

pred = kmeans.predict(finance_features)

print "k-means labels - pred: ", kmeans.labels_ - pred

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
#finance_features = np.array(finance_features)
plt.figure()
try:
    Draw(pred, finance_features[:, [0, 1]], poi, mark_poi=False, name="clusters1.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
  
"""
plt.figure()
try:
    Draw(pred, finance_features[:, [0, 2]], poi, mark_poi=False, name="clusters2.pdf", f1_name=feature_1, f2_name=feature_3)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
    
plt.figure()
try:
    Draw(pred, finance_features[:, [1, 2]], poi, mark_poi=False, name="clusters3.pdf", f1_name=feature_2, f2_name=feature_3)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
"""  