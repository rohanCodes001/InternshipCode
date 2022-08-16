#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:42:24 2022

@author: shan998
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
from datetime import datetime
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from temp import SSA
import os
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans

data = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_info.joblib')
original_time_series = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_ts.joblib')

X = data.get("x")      # Getting X ad Y locations of all the wells
Y = data.get("y")    
names = data.get("name")

keys_list = list(original_time_series.keys())
id_list = []
avg_concentrations = []
for i in keys_list:
    id_list.append(original_time_series.get(i).get('id'))
    avg_concentrations.append(np.average(original_time_series.get(i).get('ctet_conc')))

id_list.remove('299-E33-351')              # These 3 wells have nan values for a lot of time series dates.
id_list.remove('299-E33-350')
id_list.remove('299-E33-344')
del avg_concentrations[0:3]  

new_id_list = []
for i in range(0, len(id_list)):   
    new_id_list.append(id_list[i].replace("-", ""))    
  
  
X = [];
Y = [];
for i in range(0,len(new_id_list)):
    for j in range(0,len(names)):
        if new_id_list[i] == names[j]:
            X.append(data.get('x')[j])
            Y.append(data.get('y')[j])
            

fig,ax = plt.subplots(figsize = (12,8))
ax.plot(X, Y, 'o')
ax.set_xlabel("X Coordinates")
ax.set_ylabel("Y Coordinates")
ax.set_title("Current Well Locations for 200 West Area");
for i,txt in enumerate(new_id_list):
    ax.annotate(new_id_list[i], xy = (X[i], Y[i]))
    
location_X = np.array(X)
location_Y = np.array(Y)
concentrations_arr = np.array(avg_concentrations)

combined_locations = np.vstack((X,Y,concentrations_arr)).T

plt.figure(figsize=(10, 7))
plt.title("Well Locations Dendograms")
dend = shc.dendrogram(shc.linkage(combined_locations, method='ward'))  # From this we see that we have 4 clusters

kmeans = KMeans(n_clusters = 4)
kmeans.fit(combined_locations)
print(kmeans.cluster_centers_)  # The centroid values for the final clusters
print(kmeans.labels_)


fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(combined_locations[:,0], combined_locations[:,1], c = kmeans.labels_, cmap = 'rainbow')
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:,0], centroids[:,1], color = 'k')
for i,txt in enumerate(new_id_list):
    ax.annotate(new_id_list[i], xy = (X[i], Y[i]))
leg = ax.legend()