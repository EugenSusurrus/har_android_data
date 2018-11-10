# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:31:10 2018

@author: Eugen Rusu
A Spyder script to visualize the sensor data.
LINK to the GitHub Repo:
https://github.com/EugenSusurrus/har_android_data
"""

import numpy as np
import pandas as pd
from scipy import integrate
#from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Importing the data
RAW_SENSOR_DATA_500MS = pd.read_csv('har_sensor_data_500.csv', sep=',')

#Ploting the raw data
AX = plt.figure(figsize=(16, 8))
plt.plot(RAW_SENSOR_DATA_500MS['Time'], RAW_SENSOR_DATA_500MS['ACCELEROMETER X'])
plt.plot(RAW_SENSOR_DATA_500MS['Time'], RAW_SENSOR_DATA_500MS['ACCELEROMETER Y'])
plt.plot(RAW_SENSOR_DATA_500MS['Time'], RAW_SENSOR_DATA_500MS['ACCELEROMETER Z'])
plt.grid()
plt.legend()
plt.title('$ACC_X$, $ACC_Y$ and $ACC_Z$ Raw sensor data')

# Setting the working dataframe and other variables
CONSTANT_G = 9.82912

DF = pd.DataFrame({'Time': [], \
                   'ACC_X': [], \
                   'ACC_Y': [], \
                   'ACC_Z':[], \
                   'ACC_RMS': [], \
                   'Speed': [], \
                   'Distance': [], \
                   'Cumulated Distance':[]})

# Calibrating the ACC_X, ACC_Y and ACC_Z data along each accelerometer axis
DF['ACC_X'] = RAW_SENSOR_DATA_500MS['ACCELEROMETER X']
DF['ACC_Y'] = RAW_SENSOR_DATA_500MS['ACCELEROMETER Y']
DF['ACC_Z'] = RAW_SENSOR_DATA_500MS['ACCELEROMETER Z']

# Transforming the time to IS units
DF['Time'] = (RAW_SENSOR_DATA_500MS['Time'] - RAW_SENSOR_DATA_500MS['Time'].iloc[0]) / 1000 #s

# Rounds the time units
DF['Time'] = np.round(DF['Time'], decimals=2)

# The acceleration root mean square magnitude
DF['ACC_RMS'] = (DF['ACC_X']**2 + DF['ACC_Y'] **2 + DF['ACC_Z']**2) ** 0.5 - 9.81

# Speed in m/s
# Returns the cumulative integral, hence needs to be differentiated to get the instantaneous speed
DF['Speed'] = integrate.cumtrapz(DF['ACC_Z'], DF['Time'], initial=0.0)
# Differentiating to get the instantaneous speed
# df['Speed'] = df['Speed'].diff()
# Fills the first NaN value at index 0 resulted from differentiation
DF['Speed'].fillna(0, inplace=True)

# Distance in m
# Returns the cumulative integral, hence needs to be differentiated to get the
# instantaneous distance
DF['Cumulated Distance'] = integrate.cumtrapz(DF['Speed'], DF['Time'], initial=0.0)
# Diferentiating to get the instantaneous distance
# df['Distance'] = df['Cumulated Distance'].diff()
# Fills the first NaN value at the index 0 resulted from differentiation
DF['Distance'].fillna(0, inplace=True)

# Shows the resulting df
DF.head()

AX = plt.figure(figsize=(16, 8))
plt.plot(DF['Time'], DF['ACC_X'])
plt.plot(DF['Time'], DF['ACC_Y'])
plt.plot(DF['Time'], DF['ACC_Z'])
plt.plot(DF['Time'], DF['ACC_RMS'])
plt.grid()
plt.title('$ACC_X$, $ACC_Y$ and $ACC_Z$ vs $ACC_{RMS}$')
plt.legend()

# Standard deviation
AX = plt.figure(figsize=(16, 8))
plt.plot(DF['Time'], DF['ACC_RMS'])
plt.axhline(y=9.82912, color='r', linestyle=':')
plt.plot(DF['Time'], DF['ACC_RMS'].rolling(10).std())
plt.plot(DF['Time'], DF['ACC_RMS'].rolling(10).mean())
plt.legend(['ACC_RMS', 'G_TRESH', 'ACC_5S_STD', 'ACC_5S_MEAN'])
plt.grid()

AX3D = Axes3D(plt.figure(figsize=(10, 10)))

ACC_SURF = AX3D.plot_trisurf(DF['ACC_X'], DF['ACC_Y'], DF['ACC_Z'], cmap='jet', linewidth=0.1)

AX3D.scatter(DF['ACC_X'], DF['ACC_Y'], DF['ACC_Z'], color='k')

AX3D.set_xlabel('$ACC_X$')
AX3D.set_ylabel('$ACC_Y$')
AX3D.set_zlabel('$ACC_Z$')

CBAR = plt.colorbar(ACC_SURF, shrink=0.7)
CBAR.set_label('Acceleration $m/s^2$')

plt.axis('equal')
plt.title('$ACC_X$, $ACC_Y$ and $ACC_Z$ Scatter')
plt.show()

#SAcceleration RMS, Speed, Distance and Cumulated distance

FIG, AX1 = plt.subplots(figsize=(16, 8))

COLOR = 'tab:red'
AX1.set_xlabel('time (s)')
AX1.set_ylabel('ACC_Z', color=COLOR)
AX1.plot(DF['Time'], DF['ACC_Z'], color=COLOR)
AX1.tick_params(axis='y', labelcolor=COLOR)

AX2 = AX1.twinx()  # instantiate 2nd axis to plot speed
COLOR = 'tab:blue'
AX2.set_ylabel('Speed $m/s$', color=COLOR)
AX2.plot(DF['Time'], DF['Speed'], color=COLOR)
AX2.tick_params(axis='y', labelcolor=COLOR)

AX3 = AX1.twinx()  # instantiate 3rd axis to plot distance
COLOR = 'tab:green'
AX3.set_ylabel('Distance $m$', color=COLOR)
AX3.plot(DF['Time'], DF['Distance'], color=COLOR)
AX3.tick_params(axis='y', labelcolor=COLOR)

X3 = AX1.twinx()  # instantiate 3rd axis to plot distance
COLOR = 'tab:orange'
AX3.set_ylabel('Cumulated Distance $m$', color=COLOR)
AX3.plot(DF['Time'], DF['Cumulated Distance'], color=COLOR)
AX3.tick_params(axis='y', labelcolor=COLOR)

FIG.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()

# Calculates the number of steps
# Steps are considered for an ACC_RMS magnitude between 0 and 5
N_ = 5 # number of points to be checked before and after
STEP_SIGNAL = pd.DataFrame()
STEP_SIGNAL = DF[(DF['ACC_RMS'] > 0) & (DF['ACC_RMS'] <= 5)]
STEP_SIGNAL_LIST = STEP_SIGNAL['ACC_RMS'].tolist()
STEP_SIGNAL_LIST = np.array(STEP_SIGNAL_LIST)

# the number of steps
NUM_STEPS = len(STEP_SIGNAL_LIST)

# Calculate the number of jumps
# Jumps are considered for an ACC_RMS magnitude greater or equal than 6
JUMP_SIGNAL = pd.DataFrame()
JUMP_SIGNAL = DF[DF['ACC_RMS'] >= 6]
JUMP_SIGNAL_LIST = JUMP_SIGNAL['ACC_RMS'].tolist()
JUMP_SIGNAL_LIST = np.array(JUMP_SIGNAL_LIST)

# the number of jumps
NUM_JUMPS = len(JUMP_SIGNAL_LIST)

print('\n\tWelcome to the HAR app using Android sensor data\n')
print('\n\t**************************************************\n')
print('\n\tThe total distance traveled is: {} meters'.format(DF['Cumulated Distance'].max()))
print('\n\tThe average registered velocity is: {} m/s'.format((DF['Speed'].mean() ** 2) ** 0.5))
print('\n\tYou have made {} steps'.format(NUM_STEPS))
print('\n\tYou have made {} jumps'.format(NUM_JUMPS))
