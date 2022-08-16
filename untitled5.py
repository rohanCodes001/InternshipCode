#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:08:48 2022

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

os.rename("untitled5.py", "Clustering.py")

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
ax.annotate('test', xy = )