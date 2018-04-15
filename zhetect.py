#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sys

# load data (low memory: supress memory warning)
train_dp = pd.read_csv(r"tadpole/TADPOLE_D1.csv", low_memory=False)
test_dp = pd.read_csv(r"tadpole/TADPOLE_D2.csv", low_memory=False)

'''
train_data = train_dp[['MMSE','DX','DXCHANGE','AGE',
                       'PTGENDER','PTETHCAT','PTRACCAT',]]
'''

train_data = train_dp[['DX_bl','DX','MMSE',
                       'ST83TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','','',
                       '','','',
                       '','','',
                       '','','',
                       '','','',
                       '','']]

print (train_data)
