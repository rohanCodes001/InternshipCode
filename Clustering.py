#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:08:48 2022

@author: shan998
"""

# Here we are using KMeans and Hierarchical clustering approach to spatially cluster wells based on location data.

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
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

data = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_info.joblib')
original_time_series = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_ts.joblib')

X = data.get("x")      # Getting X ad Y locations of all the wells
Y = data.get("y")    
names = data.get("name")

fig,ax = plt.subplots(figsize = (12,8))
ax.plot(X, Y, 'o')
ax.set_xlabel("X Coordinates")
ax.set_ylabel("Y Coordinates")
ax.set_title("Current Well Locations for 200 West Area");
for i,txt in enumerate(names):
    ax.annotate(names[i], xy = (X[i], Y[i]))
    
combined_locations = np.vstack((X,Y)).T   # Combining both x and y locations into one array.

# Using the K means elbow plot to find the optimal number of clusters.
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(combined_locations)
    data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

kmeans = KMeans(n_clusters = 3, random_state=10)
kmeans.fit(combined_locations)
print(kmeans.cluster_centers_)  # The centroid values for the final clusters
print(kmeans.labels_)


fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(combined_locations[:,0], combined_locations[:,1], c = kmeans.labels_, cmap = 'rainbow')
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:,0], centroids[:,1], color = 'k')
for i,txt in enumerate(names):
    ax.annotate(names[i], xy = (X[i], Y[i]))
leg = ax.legend

# Silhouette score takes into consideration the intra-cluster distance between 
# the sample and other data points within the same cluster (a) and inter-cluster distance between the 
# sample and the next nearest cluster (b).
score = silhouette_score(combined_locations, kmeans.labels_, metric='euclidean') 
print(score)

#  Hierarchical Clustering with different number of clusters.
plt.figure(figsize=(10, 7))
plt.title("Well Locations Dendograms")
dend = shc.dendrogram(shc.linkage(combined_locations, method='ward'))  # From this we see that we have 5 clusters

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
cluster.fit_predict(combined_locations)

fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(combined_locations[:,0], combined_locations[:,1], c=cluster.labels_, cmap='rainbow')
score = silhouette_score(combined_locations, cluster.labels_, metric='euclidean') 
for i,txt in enumerate(names):
    ax.annotate(names[i], xy = (X[i], Y[i]))
leg = ax.legend
print(score)

