import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import learn_utils
import time
import statistics
from sklearn.cluster import KMeans
from datetime import datetime


def update_dataframe(results, trend_data, error_data, start, end, start_index, full_DF):
    total_data = full_DF.I1

    data_list = total_data[start:end]
    magnitude = abs(max(data_list, key=abs))
    
    if start - 640 > 0:
        previous_10 = []
        for i in range(10):
            previous_10.append(abs(max(total_data[start - (i+1)*64:start - (i * 64)], key = abs)))
        before = float("{:.4f}".format(np.average(previous_10)))
    else:
        before = 0 

    if end + 640 < len(total_data):
        next_10 = []
        for i in range(10):
            next_10.append(abs(max(total_data[end + (i)*64 + 1:end + ((i+1)* 64)], key = abs)))
        #next_cycle = total_data[end + 1: end + 64]
        after = float("{:.4f}".format(np.average(next_10)))
        #after = abs(max(next_cycle, key=abs))
    else:
        after = 0 

    duration = len(data_list)

    trend_value = trend.I1acAvg[(start// 3072)]
    
    error_list = error[start - start_index:end - start_index]
    size = max(error_list)

    
    DF_row = full_DF.iloc[start]
    time_str = DF_row[2] + ' ' +DF_row[3]+'.' + str(DF_row[4])
    time = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S.%f')

    label = ''
    next_row = [label, time.date(), time.time(), magnitude, before, after, duration, size, trend_value]
    results.loc[len(results)] = next_row
    return 



filename = 'Q0000016'

append = ['a', 'b', 'c', 'd']
for ending in append:

    # ***************** READ DATA *********************
    PARENT_FOLDER = r"D:\Anna\Documents\Uni\Thesis\CSV_DATA"
    RESULTS_FOLDER = r"D:\Anna\Documents\Uni\Thesis\Code\Detections"
    INPUT_PATH = os.path.join(PARENT_FOLDER, filename + ending + '.csv')
    TREND_PATH = os.path.join(PARENT_FOLDER,filename + 'trend.csv')
    RESULT_PATH = os.path.join(RESULTS_FOLDER, filename + '_results.csv')
    data = pd.read_csv(INPUT_PATH,encoding = 'unicode_escape')
    trend = pd.read_csv(TREND_PATH,encoding = 'unicode_escape') 
    results = pd.read_csv(RESULT_PATH, encoding = 'unicode_escape')

    total_length = len(data.I1)
    finalRaw = []
    finalThreshold = []
    finalError = []
    horizontalThreshold = []
    data_division = 6
    size_division = 100000
    start_index = 0 


    divided = 1
    average = 1

    currentData = data.I1
    trendData = trend.I1acAvg

    end_index =  start_index + size_division
    while end_index < total_length:
        print(end_index)

        # No. of samples
        n_plot_samples = end_index - start_index
        rawData = [0] * n_plot_samples
        plotting_data = currentData[start_index:end_index]
        if divided == 1:
            for i in range(n_plot_samples):
                rawData[i] = currentData[i + start_index] / trend.I1acAvg[((i + start_index)// 3072)]
        else:
            rawData = currentData[start_index:end_index]

        # ***************** WINDOWING *********************
        segment_len = 128
        slide_len = 10

        segments = []

        for start_pos in range (0, len(rawData), slide_len):
            end_pos = start_pos + segment_len
            # Make copy so 'segments' doesn't modify original rawData
            segment = np.copy(rawData[start_pos:end_pos])
            # If at end of list and truncated segment, drop
            if len(segment) != segment_len:
                continue
            segments.append(segment)

        #Show produced segments
        #learn_utils.plot_waves(segments, step = 1)

        #************ FORCE SEGMENTS TO START AT 0 ****************
        #Create Window
        window_rads = np.linspace(0, np.pi, segment_len)
        window = np.sin(window_rads)**2
        #plt.plot(window)
        #plt.show()

        #Window
        windowed_segments = []
        for segment in segments:
            windowed_segment = np.copy(segment)*window
            windowed_segments.append(windowed_segment)

        #learn_utils.plot_waves(windowed_segments, step = 3)


        #************ CLUSTERING ********************
        clusterer = KMeans(n_clusters = 100)
        clusterer.fit(windowed_segments)

        # ***********RECONSTRUCTION FROM CLUSTERS***************
        slide_len = int(segment_len/2)
        # Segments but slide at different length to original training (32 rather than 10)
        test_segments = learn_utils.sliding_chunker(
            rawData,
            window_len = segment_len,
            slide_len = slide_len
        )

        #Test with original waveform
        """centroids = clusterer.cluster_centers_

        segment = np.copy(test_segments[0])

        windowed_segment = segment * window
        print(windowed_segment)

        nearest_centroid_idx = clusterer.predict(windowed_segment.reshape(1, -1))[0]
        nearest_centroid = np.copy(centroids[nearest_centroid_idx])
        plt.figure()
        plt.plot(segment, label="Original segment")
        plt.plot(windowed_segment, label="Windowed segment")
        plt.plot(nearest_centroid, label="Nearest centroid")
        plt.legend()
        plt.show()"""

        #Now reconstruct the whole dataset
        reconstruction = np.zeros(len(rawData))

        #For every test segment 
        for segment_n, segment in enumerate(test_segments):
            segment = np.copy(segment)
            # Window to force start and end at 0
            segment *= window
            # Calculate nearest centroid
            nearest_centroid_idx = clusterer.predict(segment.reshape(1, -1))[0]
            centroids = clusterer.cluster_centers_
            nearest_centroid = np.copy(centroids[nearest_centroid_idx])
            # Update position
            pos = segment_n * slide_len

            # Reconstruction won't be overlapped
            reconstruction[pos:pos+segment_len] += nearest_centroid



        # Initialise error array (green)
        error_raw = np.zeros(n_plot_samples)
        error_raw[segment_len:n_plot_samples-segment_len] = abs(reconstruction[segment_len:n_plot_samples-segment_len] - rawData[segment_len:n_plot_samples-segment_len])

        #Test average error over one cycle (64 samples)
        error_zeros = np.zeros(32)
        error_ave = [np.mean(error_raw[i-32:i+32]) for i in range(32, n_plot_samples-32)] 

        if average == 1:
            error = np.concatenate([error_zeros, error_ave, error_zeros])
        else:
            error = error_raw 

        if divided == 1:
            if average == 1:
                # error_98th_percentile = 0.135
                # error_98th_percentile = np.percentile(error, 99.5)
                error_98th_percentile = statistics.median(error) + 7 * statistics.stdev(error)
            else:
                error_98th_percentile = np.percentile(error, 99)
        else:
            if average == 1:
                #error_98th_percentile = 15
                error_98th_percentile = np.percentile(error, 75)
                # error_98th_percentile = statistics.mean(error) + statistics.stddev(error)
            else:
                error_98th_percentile = np.percentile(error, 99)
        
        # Initialise error array (orange)
        threshold = plotting_data.copy()
        threshold = np.array(threshold)
        rawData = np.array(plotting_data)
        original = rawData.copy()

        # Array to keep track of which index is an error
        errorIndexArray = [0] * n_plot_samples

        # Waveform is considered error if duration longer than this threshold in samples
        sampleThreshold = 64

        # Initialise variables
        errorIndexStart = 0
        errorIndexEnd = 0
        nonErrorCounter = 0
        errorCounter = 0
        recordStart = 1
        inErrorState = 0
        erase = 0

        # Calculate when waveform is in an error state and store in errorIndexArray
        for i in range(n_plot_samples):
            # If higher than value threshold
            if error[i] > error_98th_percentile:
                threshold[i] = np.nan
                # Record index
                errorIndexEnd = i
                # If first instance, note index of the start of error
                if recordStart == 1:
                    errorIndexStart = i
                    recordStart = 0
                # Update how long error has occurred for
                errorCounter += 1   
                nonErrorCounter = 0
                # If error occurred for long enough duration, waveform is in error state
                if errorCounter >= sampleThreshold:
                    inErrorState = 1
                    # Pre-fill errorIndexArray with 1 from the start of error until now
                    for j in range (errorIndexStart, i+1):
                        threshold[j] = original[j]

            # If not higher than value threshold
            else:
                threshold[i] = np.nan
                # Update how long waveform has been normal
                nonErrorCounter += 1
                # Reset error instance
                if inErrorState == 0:
                    recordStart = 1
                    errorCounter = 0
                # If waveform has been normal for half a cycle, not in error state (anymore)
                if nonErrorCounter >= 64:
                    # Overwrite error state to be 0 from end of the last error value
                    if inErrorState == 1:
                        inErrorState = 0
                        for j in range (errorIndexEnd, i+1):
                            threshold[j] = np.nan

                        update_dataframe(results, trendData, error, errorIndexStart + start_index, errorIndexEnd + start_index, start_index, data)
                            
            # REcord in array 
            if inErrorState == 1:
                    threshold[i] = original[i]


        finalRaw = finalRaw + list(rawData)
        finalThreshold = finalThreshold + list(threshold)
        finalError = finalError + list(error)
        horizontalThreshold = horizontalThreshold + [error_98th_percentile] * len(rawData)


        if end_index + 2*size_division > total_length:
            start_index = end_index + 1
            end_index = total_length
        else:
            start_index = end_index + 1
            end_index = start_index + size_division

    results.to_csv(RESULT_PATH, index=False, header = True)  
    print("plotting")
    # Plot result
    plt.plot(horizontalThreshold, color='r', linestyle='-')
    plt.plot(finalRaw, label="Original waveform")
    plt.plot(finalThreshold, label="Anomaly")
    plt.plot(finalError, label="Reconstruction Error")
    plt.legend()
    # plt.show()

    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig(filename + ending + '.png', dpi = 100)
    plt.close()

