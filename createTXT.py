# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np

PARENT_FOLDER = r"D:\Anna\Documents\Uni\Thesis\CSV_DATA"
INPUT_PATH = os.path.join(PARENT_FOLDER,'Q0000012a.csv')
V1_OUTPUT_PATH = os.path.join(PARENT_FOLDER,'Q0000012a.csv')

print(PARENT_FOLDER)
data = pd.read_csv(INPUT_PATH,encoding = 'unicode_escape')

#Create V1.txt for RNN
np.savetxt(r'D:\Anna\Documents\Uni\Thesis\Code\Q0000012a.txt', data.I1[600000:800000], fmt='%.2f')
