import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import os
import datetime

# ***************** READ DATA *********************
#Read in raw data from .txt
with open('trend.txt') as f:
    rawData = f.readlines()
#Strip whitespace
rawData = [x.strip() for x in rawData]
#Convert string to float
rawData = [float(x) for x in rawData]


# ***************** WINDOWING *********************
segment_len = 25
slide_len = 25

segments = []

for start_pos in range (0, len(rawData), slide_len):
    end_pos = start_pos + segment_len
    # Make copy so 'segments' doesn't modify original rawData
    segment = np.copy(rawData[start_pos:end_pos])
    # If at end of list and truncated segment, drop
    if len(segment) != segment_len:
        continue
    segments.append(segment)
print("Produced %d waveform segments" % len(segments))

gradients = [0] * len(segments)
yIntercept = [0] * len(segments)
#print(segments[0])

for count,segment in enumerate(segments):
    indices = np.zeros(segment_len)
    dataFit = np.zeros(segment_len)
    for index, data in enumerate(segment):
        indices[index] = index
        dataFit[index] = data
    m, c = np.polyfit(indices, dataFit, 1)
    gradients[count] = m
    yIntercept[count] = c

#print(gradients)

reconstruction = np.zeros(len(rawData))
x = np.linspace(0, segment_len-1, segment_len)
y = np.linspace(0, segment_len-1, segment_len)
print(x)
for i in range(0, len(rawData), segment_len):
    y = gradients[int(i/segment_len)] * x + yIntercept[int(i/segment_len)]
    #print(y)
    reconstruction[i:i+segment_len]= y

reconstructionError = np.copy(reconstruction)


# If value in errorIndexArray = 1, plot error graph, else plot normal waveform (different colours)
for i in range(0, len(rawData), segment_len):
    # If error, remove from normal waveform
    if abs(gradients[int(i/segment_len)]) > 2:
        reconstruction[i:i+segment_len-1] = np.nan
    # If not error, remove from error waveform
    else:
        reconstructionError[i:i+segment_len-1] = np.nan

plt.plot(rawData[0:len(rawData)], label="Original waveform")
plt.plot(reconstruction[0:len(rawData)], label = "gradient reconstruction")
plt.plot(reconstructionError[0:len(rawData)], label = "gradient error")
plt.legend()
plt.show()