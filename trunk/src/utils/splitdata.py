#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    context = os.getcwd().split('\\')
else:
    # Ubuntu
    context = os.getcwd().split('/')

context = '/'.join(context[:context.index('AlzheTect') + 1])

def SplitTrainTestData(indata=context + r"/trunk/src/tadpole/TADPOLE_D1_D2.csv", trainfile=context + r"/trunk/src/train/TADPOLE_D12.csv", testfile=context + r"/trunk/src/test/TADPOLE_D22.csv"):

    # input data from tadpole mixed file and separate into training and testing
    tadpole_dp = pd.read_csv(indata, low_memory=False)

    d1_data = tadpole_dp.loc[tadpole_dp['D1'] == 1]
    d2_data = tadpole_dp.loc[tadpole_dp['D2'] == 1]

    d1_data.to_csv(trainfile, encoding='utf-8', index=False)
    d2_data.to_csv(testfile, encoding='utf-8', index=False)

    print ('Total data shape:', tadpole_dp.shape)
    print ('Train data shape:', d1_data.shape)
    print ('Test data shape:', d2_data.shape)

    return d1_data, d2_data

def SplitDXData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    cn_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "CN"]
    ad_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "AD"]

    print ('Total data shape:', tadpole_dp.shape)
    return cn_data, ad_data

def SplitDXDataCSV(indata=context + r"/trunk/src/train/TADPOLE_D1.csv", cnfile=context + r"/trunk/src/train/TADPOLE_D1_CN.csv", adfile=context + r"/trunk/src/train/TADPOLE_D1_AD.csv"):
    cn_data, ad_data = SplitDXData(indata=indata)

    cn_data.to_csv(cnfile, encoding='utf-8', index=False)
    ad_data.to_csv(adfile, encoding='utf-8', index=False)

    print ('CN data shape:', cn_data.shape)
    print ('AD data shape:', ad_data.shape)

def SplitClassData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    cn_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "CN"]
    ad_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "AD"]
    c_data = pd.concat([ad_data,cn_data])

    print ('Total data shape:', tadpole_dp.shape)
    return c_data

if __name__ == "__main__":

    train_cn_data, train_ad_data = SplitDXData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv")
    test_cn_data, test_ad_data = SplitDXData(indata=context + r"/trunk/src/test/TADPOLE_D2.csv")

    print ('Train CN data shape:', train_cn_data.shape)
    print ('Train AD data shape:', train_ad_data.shape)
    print ('Test CN data shape:', test_cn_data.shape)
    print ('Test AD data shape:', test_ad_data.shape)
