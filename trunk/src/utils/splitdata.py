#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

from datetime import datetime

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    context = os.getcwd().split('\\')
else:
    # Ubuntu
    context = os.getcwd().split('/')

context = '/'.join(context[:context.index('AlzheTect') + 1])

def SplitTrainTestData(indata=context + r"/trunk/src/tadpole/TADPOLE_D1_D2.csv", trainfile=context + r"/trunk/src/train/TADPOLE_D1.csv", testfile=context + r"/trunk/src/test/TADPOLE_D2.csv"):

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

def SplitTrainTestData2(indata=context + r"/trunk/src/tadpole/TADPOLE_D1_D2.csv", trainfile=context + r"/trunk/src/train/TADPOLE_train.csv", testfile=context + r"/trunk/src/test/TADPOLE_test.csv"):

    # input data from tadpole mixed file and separate into training and testing
    tadpole_dp = pd.read_csv(indata, low_memory=False)

    data = {}
    final_data = []
    # print tadpole_dp.head()
    rids = tadpole_dp['RID'].values.tolist()
    rids = set(rids)
    print len(rids)
    for id in rids:
        data[id] = tadpole_dp.loc[tadpole_dp['RID'] == id]

    print len(data)
    # print data.keys()

    for key, value in data.iteritems():
        max_obj = None
        for date in value['EXAMDATE'].values:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if max_obj is None or date_obj > max_obj[0]:
                max_obj = [date_obj, value.loc[value['EXAMDATE'] == date]]
        # print 'max type:', type(max_obj[1])
        # print max_obj[1]
        final_data.append(max_obj[1])
        # print 'final type', type(final_data)
        # print final_data
        # break

    final_data = pd.concat(final_data)
    final_data = final_data.sample(frac=1)
    print 'final data shape:', final_data.shape
    print 'final data len:', len(final_data)
    print ('Total data shape:', tadpole_dp.shape)

    final_data['DX_bl'].replace(['LMCI', 'EMCI', 'SMC'], ['MCI', 'MCI', 'MCI'], inplace=True)

    train_index = int(len(final_data) * 0.8)

    d1_data = final_data[:train_index]
    d2_data = final_data[train_index:]

    d1_data.to_csv(trainfile, encoding='utf-8', index=False)
    d2_data.to_csv(testfile, encoding='utf-8', index=False)

    # final_data.to_csv(trainfile, encoding='utf-8', index=False)


def SplitDXData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    cn_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "CN"]
    ad_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "AD"]
    mci_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "MCI"]
    print ('Total data shape:', tadpole_dp.shape)
    return cn_data, ad_data, mci_data

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
    mci_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "MCI"]
    # print mci_data.head()
    # c_data = pd.concat([ad_data,cn_data, mci_data])
    c_data = pd.concat([ad_data,cn_data])

    print ('Total data shape:', tadpole_dp.shape)
    return c_data
    # return tadpole_dp

if __name__ == "__main__":

    train_cn_data, train_ad_data, train_mci_data = SplitDXData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv")
    test_cn_data, test_ad_data, test_mci_data = SplitDXData(indata=context + r"/trunk/src/test/TADPOLE_D2.csv")

    print ('Train CN data shape:', train_cn_data.shape)
    print ('Train AD data shape:', train_ad_data.shape)
    print ('Test CN data shape:', test_cn_data.shape)
    print ('Test AD data shape:', test_ad_data.shape)
