import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pandas as pd
import matplotlib.ticker as ticker

 ### Plot datetime on x axis

PARENT_FOLDER = os.path.abspath(os.path.join(__file__ ,".."))
#PARENT_FOLDER = 'E:\Anna\CSV_DATA'
INPUT_PATH = os.path.join(PARENT_FOLDER,'trend.csv')
#V1_OUTPUT_PATH = os.path.join(PARENT_FOLDER,'trend.csv')

"""f12a = os.path.join(PARENT_FOLDER,'Q0000012a.csv')
f12b = os.path.join(PARENT_FOLDER,'Q0000012b.csv')
f12c = os.path.join(PARENT_FOLDER,'Q0000012c.csv')
f12d = os.path.join(PARENT_FOLDER,'Q0000012d.csv')"""

#print(PARENT_FOLDER)
data = pd.read_csv(INPUT_PATH,encoding = 'unicode_escape')      

xAxisplot = data.Time 

# ***************** READ DATA *********************
#Read in raw data from .txt
with open('trend.txt') as f:
    rawData = f.readlines()
#Strip whitespace
rawData = [x.strip() for x in rawData]
#Convert string to float
rawData = [float(x) for x in rawData]


# ***************** WINDOWING *********************
segment_len = 20
slide_len = 5

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

indices = np.zeros(len(rawData))

for count,segment in enumerate(segments):
    segmentMax = max(segment)
    segmentMin = min(segment)
    currentIndex = count*slide_len
    if abs(segmentMax - segmentMin) >= 50:
        indices[currentIndex: currentIndex + segment_len] = 1

reconstruction = np.copy(rawData)


# Initialise variables
errorIndexStart = 0
errorIndexEnd = 0
nonErrorCounter = 0
errorCounter = 0
recordStart = 1
inErrorState = 0

# Calculate when waveform is in an error state and store in errorIndexArray
for i in range(0, len(rawData)):
    # If higher than value threshold
    if indices[i] == 1 :
        # Record index
        errorIndexEnd = i
        inErrorState = 1
        # If first instance, note index of the start of error
        if recordStart == 1:
            errorIndexStart = i
            recordStart = 0
        # Update how long error has occurred for
        errorCounter += 1   
        nonErrorCounter = 0
 
    # If not higher than value threshold
    else:
        # Update how long waveform has been normal
        nonErrorCounter += 1
        # Reset error instance
        recordStart = 1
        errorCounter = 0
        # If waveform has been normal for a cycle, not in error state (anymore)
        if nonErrorCounter >= 75:
            inErrorState = 0
            # Overwrite error state to be 0 from end of the last error value
            for j in range (errorIndexEnd, i):
                indices[j] =  0
    if inErrorState == 1:
        indices[i] = 1


# If value in errorIndexArray = 1, plot error graph, else plot normal waveform (different colours)
for i in range(0, len(rawData)):
    # If error, remove from normal waveform
    if indices[i] == 0:
        reconstruction[i] = np.nan

startLength = 0
endLength = 4000

plt.plot(xAxisplot.values[startLength:endLength],rawData[startLength:endLength], label="Original waveform")
plt.plot(xAxisplot.values[startLength:endLength],reconstruction[startLength:endLength], label = "gradient reconstruction")
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator((endLength - startLength)/10))
plt.legend()
plt.show()