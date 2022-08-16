#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:21:22 2022

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


data = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_info.joblib')
original_time_series = joblib.load('/Users/shan998/Downloads/Rohan_data/Week1/well_ts.joblib')


keys_list = list(original_time_series.keys())
#for i in original_time_series.keys(): 
num = 21   # This number controls which well concentration time series data to use
current_well_id =  keys_list[num]
time_series_dates = original_time_series.get(keys_list[num]).get('times')
well_concentration = original_time_series.get(keys_list[num]).get('ctet_conc')
well_concentration_mass = original_time_series.get(keys_list[num]).get('ctet_mass')
aqueous = original_time_series.get(keys_list[num]).get('aqueous')

new_dates = pd.to_datetime(time_series_dates)

# %%
fig,ax = plt.subplots(figsize = (12,8))
ax.plot(new_dates,well_concentration)  # if we want dates on the original time series plot uncomment this line
#well_concentration.plot(xlim = new_dates)
ax.set_xlabel("Dates")
ax.set_ylabel('CT Concentration Values (ug/L)')
ax.set_title(current_well_id + " Original Time Series Plot");

# %%
start = 0 # start date.
end = 101 # last date. 
window = 40 # 2 <= L <= N/2 #number of samples used for window length.
ssa = SSA(well_concentration.loc[start:end], window)
df = ssa.components_to_df(n = 20)
df.set_index(new_dates, inplace = True)
joblib.dump(df, '299-W11-49-Concentration.joblib', compress=4)

# %%
# Here 2 groups seem to be seen (0 to 20) and (20-40).
ssa.calc_wcorr()
ssa.plot_wcorr()
plt.title("W-Correlation for well concentration time series for well " + current_well_id);

# %% Here we are comparing components to the orginal time series to see periodicities and trends.
# We can F0 is the trend while f1 and f2 are other periodicties.
ssa.reconstruct(0).plot()
ssa.reconstruct([1,2]).plot()
ssa.reconstruct([3,4]).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("Concentration Time Series: Trend and Periodicities")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Concentrations")
legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(3)] + ["Original TS"]
plt.legend(legend);

# %% Looking at the first 5 elementary components together in comparison to og time series.
ssa.reconstruct(slice(0,5)).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("Concentration Time Series: Low-Frequency Periodicity (First 5 Components)")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Concentrations")
plt.legend(['Components', "Original TS"])

# %%Looking at the next components of the time series (5 - 11).
ssa.reconstruct([0] + [i for i in range(5,11)]).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("Concentration Time Series: Higher-Frequency Periodicity (Components 5-11)")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Concentrations")
plt.legend(['Components', "Original TS"])

# %%
# Looking at First 20 elements and showing the last elements since 40 was our window size.
ssa.reconstruct(slice(0,20)).plot()
ssa.reconstruct(slice(20,40)).plot()
ssa.orig_TS.plot(alpha=0.5, color = 'black')
plt.legend(["First 20 Components", "Remaining Components", "Original TS"]);
plt.title("Original Time Series versus Reconstructed Time series.")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Concentrations")

######################################################################################## Now we have mass next.

# %%
fig,ax = plt.subplots(figsize = (12,8))
ax.plot(new_dates,well_concentration_mass)  # if we want dates on the original time series plot uncomment this line
#well_concentration.plot(xlim = new_dates)
ax.set_xlabel("Dates")
ax.set_ylabel('CT Mass Values (Kg)')
ax.set_title(current_well_id + " Original Time Series Plot");

# %%
start = 0 # start date.
end = 101 # last date. 
window = 40 # 2 <= L <= N/2 #number of samples used for window length.
ssa = SSA(well_concentration_mass.loc[start:end], window)
df = ssa.components_to_df(n = 25)
df.set_index(new_dates, inplace = True)
joblib.dump(df, '299-W11-49-CTMass.joblib', compress=4)

# %%
# Here 2 groups seem to be seen (0 to 25) and (25-40).
ssa.plot_wcorr()
plt.title("W-Correlation for well mass time series for well " + current_well_id);

# %% Here we are comparing components to the orginal time series to see periodicities and trends.
# We can F0 is the trend while f1 and f2 are other periodicties.
ssa.reconstruct(0).plot()
ssa.reconstruct([1,2]).plot()
ssa.reconstruct([3,4]).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title('CT Mass Time Series: Trend and Periodicities')
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Mass")
legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(3)] + ["Original TS"]
plt.legend(legend);

# %% Looking at the first 5 elementary components together in comparison to og time series.
ssa.reconstruct(slice(0,5)).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("CT Mass Time Series: Low-Frequency Periodicity (First 5 Components)")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Mass")
plt.legend(['Components', "Original TS"])

# %% Looking at the next components of the time series (5 - 11).
ssa.reconstruct([0] + [i for i in range(5,11)]).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("CT Mass Time Series: Higher-Frequency Periodicity (Components 5-11)")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Mass")
plt.legend(['Components', "Original TS"])

# %% Looking at First 25 elements and showing the last elements since 40 was our window size.
ssa.reconstruct(slice(0,25)).plot()
ssa.reconstruct(slice(25,40)).plot()
ssa.orig_TS.plot(alpha=0.5, color = 'black')
plt.legend(["First 25 Components", "Remaining Components", "Original TS"]);
plt.title("Original Time Series versus Reconstructed Time series.")
plt.xlabel("Number of Records in Time")
plt.ylabel("CT Mass (Kg)")


####################################################################################### Now we have aqueous mass next.

# %%
fig,ax = plt.subplots(figsize = (12,8))
ax.plot(new_dates, abs(aqueous))  # if we want dates on the original time series plot uncomment this line
#well_concentration.plot(xlim = new_dates)
ax.set_xlabel("Dates")
ax.set_ylabel('Aqueous Mass Values (GPM)')
ax.set_title(current_well_id + " Original Time Series Plot");

# %%
start = 0 # start date.
end = 101 # last date. 
window = 40 # 2 <= L <= N/2 #number of samples used for window length.
ssa = SSA(abs(aqueous.loc[start:end]), window)
df = ssa.components_to_df(n = 33)
df.set_index(new_dates, inplace = True)
joblib.dump(df, '299-W11-49-CTAqMass.joblib', compress=4)

# %%
ssa.plot_wcorr()
plt.title("W-Correlation for aqueous mass time series for well " + current_well_id);

# %% Here we are comparing components to the orginal time series to see periodicities and trends.
# We can F0 is the trend while f1 and f2 are other periodicties.
ssa.reconstruct(0).plot()
ssa.reconstruct([1,2]).plot()
ssa.reconstruct([3,4]).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title('Aqueous Mass Time Series: Trend and Periodicities')
plt.xlabel("Number of Records in Time")
plt.ylabel("Aqueous Mass")
legend = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(3)] + ["Original TS"]
plt.legend(legend);

# %% Looking at the first 5 elementary components together in comparison to og time series.
ssa.reconstruct(slice(0,5)).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("Aqueous Mass Time Series: Low-Frequency Periodicity (First 5 Components)")
plt.xlabel("Number of Records in Time")
plt.ylabel("Aqueous Mass")
plt.legend(['Components', "Original TS"])

# %% Looking at the next components of the time series (5 - 11).
ssa.reconstruct([0] + [i for i in range(5,11)]).plot()
ssa.orig_TS.plot(alpha=0.4, color = 'black')
plt.title("Aqueous Mass Time Series: Higher-Frequency Periodicity (Components 5-11)")
plt.xlabel("Number of Records in Time")
plt.ylabel("Aqueous Mass")
plt.legend(['Components', "Original TS"])
    
# %% Looking at First 33 elements and showing the last elements since 40 was our window size.
ssa.reconstruct(slice(0,33)).plot()
ssa.reconstruct(slice(33,40)).plot()
ssa.orig_TS.plot(alpha=0.5, color = 'black')
plt.legend(["First 33 Components", "Remaining Components", "Original TS"]);
plt.title("Original Time Series versus Reconstructed Time series.")
plt.xlabel("Number of Records in Time")
plt.ylabel("Aqueous Mass (GPM)")