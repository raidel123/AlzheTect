#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sys
import os

cwd = os.getcwd()
print (cwd)
sys.path.append(cwd + "/utils")

import splitdata as sd

# load data (low memory: supress memory warning)
train_dp = pd.read_csv(r"tadpole/TADPOLE_D1.csv", low_memory=False)
test_dp = pd.read_csv(r"tadpole/TADPOLE_D2.csv", low_memory=False)

'''
train_data = train_dp[['MMSE','DX','DXCHANGE','AGE',
                       'PTGENDER','PTETHCAT','PTRACCAT',]]
'''

train_data = train_dp[['DX_bl','DX',
                       'MMSE', 'MMSE_bl', 'ADAS11', 'ADAS13','CDRSB', 'RAVLT_immediate',
                       'Hippocampus','WholeBrain','Entorhinal', 'MidTemp',
                       'FDG','AV45',
                       'ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17',
                       'APOE4','AGE','PTGENDER',
                       'ST83TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST83TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST83CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                       'ST24TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST24TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST24CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST113CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST54CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST39TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST39TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                       'ST39SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV1451_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV1451_10_17_16',
                       'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV1451_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV1451_10_17_16',
                       'ST89SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST130TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST130TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                       'ST12SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_UCBERKELEYAV1451_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV1451_10_17_16',
                       'ST40TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST40TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST117SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST109TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST110TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST110TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       ]]

test_data = test_dp[['DX_bl','DX',
                       'MMSE', 'MMSE_bl', 'ADAS11', 'ADAS13','CDRSB', 'RAVLT_immediate',
                       'Hippocampus','WholeBrain','Entorhinal', 'MidTemp',
                       'FDG','AV45',
                       'ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17',
                       'APOE4','AGE','PTGENDER',
                       'ST83TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST83TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST83CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                       'ST24TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST24TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST24CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST113CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST54CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST39TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST39TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                       'ST39SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV1451_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV1451_10_17_16',
                       'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV1451_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV1451_10_17_16',
                       'ST89SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST130TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST130TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                       'ST12SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_UCBERKELEYAV1451_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV1451_10_17_16',
                       'ST40TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST40TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST117SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST109TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       'ST110TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST110TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                       ]]

CN_train_data, AD_train_data = sd.SplitDXData(indata=train_data, file=False)
CN_test_data, AD_test_data = sd.SplitDXData(indata=test_data, file=False)

print ('CN train data shape:', CN_train_data.shape)
print ('AD train data shape:', AD_train_data.shape)

print ('CN test data shape:', CN_test_data.shape)
print ('AD test data shape:', AD_test_data.shape)
