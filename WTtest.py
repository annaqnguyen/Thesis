# -*- coding: utf-8 -*-
import pandas as pd
import os
import luminol
from luminol.anomaly_detector import AnomalyDetector
import matplotlib.pyplot as plt 
import pywt
import pywt.data
import numpy as np
import csv

PARENT_FOLDER = os.path.abspath(os.path.join(__file__ ,".."))
INPUT_PATH_CURRENT = os.path.join(PARENT_FOLDER,'Q0000011_anom.csv')
INPUT_PATH = os.path.join(PARENT_FOLDER,'Q0000011.csv')
I4_OUTPUT_PATH = os.path.join(PARENT_FOLDER,'I4.csv')
V1_OUTPUT_PATH = os.path.join(PARENT_FOLDER,'V1.csv')

data = pd.read_csv(INPUT_PATH,encoding = 'unicode_escape',parse_dates={'Datetime': [2,3,4]})
dataC = pd.read_csv(INPUT_PATH_CURRENT,encoding = 'unicode_escape',parse_dates={'Datetime': [2,3,4]})

#Create current CSV
#Time | I1 
i4 = pd.DataFrame({'Datetime': dataC.Datetime, 'I4': dataC.I4})
i4.to_csv(I4_OUTPUT_PATH,index=False)

#Create voltage CSV
#Time | V1 
#v1 = pd.DataFrame({'Datetime': data.Datetime, 'V1': data.V1})
#v1.to_csv(V1_OUTPUT_PATH,index=False)

"""x = np.arange(15360)
y = data.V1
coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
plt.matshow(coef) # doctest: +SKIP
plt.show() # doctest: +SKIP"""
"""x = np.arange(512)
y = np.sin(2*np.pi*x/32)
coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
plt.matshow(coef) # doctest: +SKIP
plt.show() # doctest: +SKIP"""


x = np.linspace(0, 1, num=2048)
chirp_signal = np.sin(250 * np.pi * x**2)
    
fig, ax = plt.subplots(figsize=(6,1))
ax.set_title("Original Chirp Signal: ")
ax.plot(chirp_signal)
#plt.show()
    
#test = data.V1[0:2000]
test = dataC.I1[12000:]
#test = chirp_signal
waveletname = 'sym5'

fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
for ii in range(5):
    (test, coeff_d) = pywt.dwt(test, waveletname)
    axarr[ii, 0].plot(test, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    '''if ii == 4:
        np.savetxt("transformed.csv", test, delimiter=",")'''
        
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
    
plt.tight_layout()
plt.show()