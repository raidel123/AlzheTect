#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# load data (low memory: supress memory warning)
train_dp = pd.read_csv(r"tadpole/TADPOLE_D1.csv", low_memory=False)
test_dp = pd.read_csv(r"tadpole/TADPOLE_D2.csv", low_memory=False)

'''
train_data = train_dp[['MMSE','DX','DXCHANGE','AGE',
                       'PTGENDER','PTETHCAT','PTRACCAT',]]
'''

train_data = train_dp[['MMSE','','',
                       '','','',
                       '','','',
                       '','','',
                       '','','',
                       '','','',
                       '','']]

print (train_data)


'''
# input data from tadpole mixed file and separate into training and testing
# test_data = pd.read_csv(r"../input/test.csv")

#print (tadpole_dp.D1.unique())
#print (tadpole_dp.D2.unique())

#train_data = tadpole_dp.loc[tadpole_dp['D1'] == 1]
#test_data = tadpole_dp.loc[tadpole_dp['D2'] == 1]

# train_data.to_csv("tadpole/TADPOLE_D1.csv", encoding='utf-8', index=False)
# test_data.to_csv("tadpole/TADPOLE_D2.csv", encoding='utf-8', index=False)

print (tadpole_dp.shape)
'''
