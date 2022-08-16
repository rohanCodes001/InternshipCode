#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:49:25 2022

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
import math as math
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import davies_bouldin_score

directory = '/Users/shan998/.spyder-py3/'

well_one = joblib.load('299-W5-1-CTAqMass.joblib')            # Loading Data
well_two = joblib.load('299-W6-15-CTAqMass.joblib')
well_three = joblib.load('299-W11-49-CTAqMass.joblib')
well_four = joblib.load('299-W11-50-CTAqMass.joblib')
well_five = joblib.load('299-W11-90-CTAqMass.joblib')
well_six = joblib.load('299-W11-92-CTAqMass.joblib')
well_seven = joblib.load('299-W11-96-CTAqMass.joblib')
well_eight = joblib.load('299-W11-97-CTAqMass.joblib')
well_nine = joblib.load('299-W12-2-CTAqMass.joblib')
well_ten = joblib.load('299-W12-3-CTAqMass.joblib')
well_eleven = joblib.load('299-W12-4-CTAqMass.joblib')  
well_twelve = joblib.load('299-W14-20-CTAqMass.joblib')
well_thirteen = joblib.load('299-W14-21-CTAqMass.joblib')
well_fourteen = joblib.load('299-W14-22-CTAqMass.joblib')
well_fifteen = joblib.load('299-W14-73-CTAqMass.joblib')
well_sixteen = joblib.load('299-W14-74-CTAqMass.joblib')
well_seventeen = joblib.load('299-W15-225-CTAqMass.joblib')
well_eighteen = joblib.load('299-W17-2-CTAqMass.joblib')
well_nineteen = joblib.load('299-W17-3-CTAqMass.joblib')
original_time_series = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_ts.joblib')
data = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_info.joblib')

#columns_list = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13',
               # 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23' , 'F24']   
myData = []
namesofmyData = []
dates = []
for filename in os.listdir(directory):
    if filename.endswith("CTAqMass.joblib"):
        df = joblib.load(directory+filename)
        df = pd.DataFrame(df.loc[:,df.columns != 'F0'].sum(axis = 1), columns = ['Sum of Components'])
        #df = df.loc[:,["F0"]]       #Enable this to look at individual trend components only.
        myData.append(df)
        namesofmyData.append(filename[:-7])
        dates.append(df.index)
        
fig, axs = plt.subplots(5,4,figsize=(25,25))
fig.suptitle('Time Series Components for CTET Aqueous Mass', fontsize = 35)
for i in range(6):
    for j in range(4):
        if i*4+j+1>len(myData): 
            continue
        axs[i, j].plot(dates[0], myData[i*4+j].values)
        axs[i, j].set_title(namesofmyData[i*4+j])
plt.show()

series_lengths = {len(series) for series in myData}
print(series_lengths)

stats_list = []
for i in range(0, len(myData)):
    temp = myData[i].describe()        # Getting stats of the summed components for each of the 19 wells.
    temp.rename(columns = {'Sum of Components': 'Sum of ' + namesofmyData[i] + ' Components'}, inplace = True)
    stats_list.append(temp)

for i in range(len(myData)):        # Normalizing values for convention purposes.
    scaler = MinMaxScaler()
    myData[i] = MinMaxScaler().fit_transform(myData[i])
    myData[i]= myData[i].reshape(len(myData[i]))
    
# Using the K means elbow plot to find the optimal number of clusters.
sse = {}
for k in range(1, 10):
    km = TimeSeriesKMeans(n_clusters=k, metric = "dtw", max_iter=1000, dtw_inertia=True, random_state=21).fit(myData)
    data["clusters"] = km.labels_
    sse[k] = km.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title("K-Means Elbow Plot")
plt.show()    

cluster_count = 4   # Optimal amount of clusters is 4 for looking at all periodic components.
                    # For trend component it is 3 clusters.
                    
km = TimeSeriesKMeans(n_clusters = cluster_count, metric="dtw", random_state=21)  # Model is here

labels = km.fit_predict(myData)
 
s_x = s_y = math.floor(math.sqrt(math.sqrt(len(myData))))   

plot_count = math.ceil(math.sqrt(cluster_count))             

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))     # Plot of the clusters from time series POV.
fig.suptitle('Clusters for Time Series Components for CTET Aqueous Mass', fontsize = 35)
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(myData[i],c="green",alpha=0.4)
                cluster.append(myData[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="black")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*s_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0     
plt.show()

# Barplot visulization of the clusters.
cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
cluster_n = ["Cluster "+str(i) for i in range(cluster_count)]
plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()

fancy_names_for_labels = [f"Cluster {label}" for label in labels]
table = pd.DataFrame(zip(namesofmyData,fancy_names_for_labels),columns=["Wells","Cluster"]).sort_values(by="Cluster").set_index("Wells")

X = data.get("x")      # Getting X and Y locations of all the wells
Y = data.get("y")    
well_name = data.get("well")

keys_list = list(original_time_series.keys())
id_list = []
for i in keys_list:
    id_list.append(original_time_series.get(i).get('id'))
  
# Getting X and y locations of wells in our selected wells dataset. 
X = [];
Y = [];
for i in range(0,len(namesofmyData)):
    for j in range(0,len(well_name)):
        if namesofmyData[i].replace('-CTAqMass', '') == well_name[j]:
            X.append(data.get('x')[j])
            Y.append(data.get('y')[j])

fig,ax = plt.subplots(figsize = (12,8))
ax.plot(X, Y, 'o')
ax.set_xlabel("X Coordinates")
ax.set_ylabel("Y Coordinates")
ax.set_title("Current Selected Well Locations for 200 West Area");
for i,txt in enumerate(namesofmyData):
    ax.annotate(namesofmyData[i].replace('-CTAqMass', ''), xy = (X[i], Y[i]))    

    
# Spatially Plotting Cluster locations.  
combined_locations = np.vstack((X,Y)).T
fig,ax = plt.subplots(figsize = (12,8))
scatter = ax.scatter(combined_locations[:,0], combined_locations[:,1], c = km.labels_, cmap = 'rainbow')
ax.set_title("Well Locations labeled with clusters using CTET Aqueous Mass data")
ax.set_xlabel("X Coordinates")
ax.set_ylabel("Y Coordinates")
for i,txt in enumerate(namesofmyData):
    ax.annotate(namesofmyData[i].replace('-CTAqMass', ''), xy = (X[i], Y[i]))
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Clusters")
ax.add_artist(legend1)    
ax.legend()
joblib.dump(combined_locations, 'ClusteringAqMass_Locations.joblib', compress=4)
joblib.dump(labels, 'ClusterAqMass_Labels.joblib', compress=4)
joblib.dump(namesofmyData, 'CTAqMasswell_names.joblib', compress=4)

# The silhouette score of 1 means that the clusters are very dense and nicely separated. 
# The score of 0 means that clusters are overlapping. 
# The score of less than 0 means that data belonging to clusters may be wrong/incorrect.
score = silhouette_score(myData, km.labels_, metric="dtw")  

# The Davies–Bouldin index (DBI) (introduced by David L. Davies and Donald W. Bouldin in 1979), 
# a metric for evaluating clustering algorithms, is an internal evaluation scheme, where the validation 
# of how well the clustering has been done is made using quantities and features inherent to the dataset. 
# Typically the lower the DB index value, better is the clustering. 
print(davies_bouldin_score(myData, km.labels_))
  