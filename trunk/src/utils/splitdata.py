#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import math
from datetime import datetime
from datetime import timedelta

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    context = os.getcwd().split('\\')
else:
    # Ubuntu
    context = os.getcwd().split('/')

context = '/'.join(context[:context.index('AlzheTect') + 1])

def SplitTrainTestDataDX(indata=context + r"/trunk/src/tadpole/TADPOLE_D1_D2.csv", trainfile=context + r"/trunk/src/train/TADPOLE_train_MCI.csv", testfile=context + r"/trunk/src/test/TADPOLE_test_MCI.csv"):

    # input data from tadpole mixed file and separate into training and testing
    tadpole_dp = pd.read_csv(indata, low_memory=False)

    data = {}
    CN_data = []
    MCI_data = []
    AD_data = []
    MCI_AD_data = []
    # print tadpole_dp.head()
    rids = tadpole_dp['RID'].values.tolist()
    rids = set(rids)
    # print len(rids)
    for id in rids:
        data[id] = tadpole_dp.loc[tadpole_dp['RID'] == id]

    for key, value in data.iteritems():
        prev_obj = None
        for index, row in value.iterrows():
            date = row['EXAMDATE']
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if prev_obj is None and row['DXCHANGE'] == 1:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 1]
            elif prev_obj is not None and row['DXCHANGE'] == 1:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 1]
            elif math.isnan(row['DXCHANGE']):
                continue
            else:
                prev_obj = None
                break

        if prev_obj is not None:
            CN_data.append(prev_obj[1])

    print "len CN:", len(CN_data)
    CN_data = pd.concat(CN_data)

    for key, value in data.iteritems():
        prev_obj = None
        for index, row in value.iterrows():
            date = row['EXAMDATE']
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if prev_obj is None and row['DXCHANGE'] == 2:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 2]
            elif prev_obj is not None and row['DXCHANGE'] == 2:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 2]
            elif math.isnan(row['DXCHANGE']):
                continue
            else:
                prev_obj = None
                break

        if prev_obj is not None:
            MCI_data.append(prev_obj[1])

    print "len MCI:", len(MCI_data)
    MCI_data = pd.concat(MCI_data)

    for key, value in data.iteritems():
        prev_obj = None
        for index, row in value.iterrows():
            date = row['EXAMDATE']
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if prev_obj is None and row['DXCHANGE'] == 3:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 3]
            elif prev_obj is not None and row['DXCHANGE'] == 3:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 3]
            elif math.isnan(row['DXCHANGE']):
                continue
            else:
                prev_obj = None
                break

        if prev_obj is not None:
            AD_data.append(prev_obj[1])

    print "len AD:", len(AD_data)
    AD_data = pd.concat(AD_data)

    for key, value in data.iteritems():
        prev_obj = None
        for index, row in value.iterrows():
            date = row['EXAMDATE']
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if prev_obj is None and row['DXCHANGE'] == 2:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 2]
            elif prev_obj is not None and row['DXCHANGE'] == 2:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 2]
            elif prev_obj is not None and (row['DXCHANGE'] == 3 or row['DXCHANGE'] == 5):
                # max_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 2]
                # if prev_obj[1]['DXCHANGE'] == 2
                    # prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date]]
                MCI_AD_data.append(prev_obj[1])
                break
    print "len MCI_AD:", len(MCI_AD_data)
    MCI_AD_data = pd.concat(MCI_AD_data)
    # final_data['DX_bl'].replace(['LMCI', 'EMCI', 'SMC'], ['MCI', 'MCI', 'MCI'], inplace=True)
    MCI_AD_data['DXCHANGE'].replace([2], [5], inplace=True)

    # final_data = pd.concat(CN_data, MCI_data, AD_data)
    final_data = CN_data.copy()
    final_data = final_data.append(MCI_data.copy())
    final_data = final_data.append(AD_data.copy())
    final_data = final_data.append(MCI_AD_data.copy())
    final_data = final_data.sample(frac=1)
    print final_data.head()

    final_data['DX_bl'].replace(['LMCI', 'EMCI', 'SMC'], ['MCI', 'MCI', 'MCI'], inplace=True)
    # print "Unique:", final_data.DXCHANGE.unique

    train_index = int(len(final_data) * 0.8)

    d1_data = final_data[:train_index]
    d2_data = final_data[train_index:]

    d1_data.to_csv(trainfile, encoding='utf-8', index=False)
    d2_data.to_csv(testfile, encoding='utf-8', index=False)

    # final_data.to_csv(trainfile, encoding='utf-8', index=False)

def SplitTrainTestDataTime(indata=context + r"/trunk/src/tadpole/TADPOLE_D1_D2.csv", trainfile=context + r"/trunk/src/train/TADPOLE_train_time.csv", testfile=context + r"/trunk/src/test/TADPOLE_test_time.csv"):

    # input data from tadpole mixed file and separate into training and testing
    tadpole_dp = pd.read_csv(indata, low_memory=False)

    data = {}
    MCI_AD_data = []
    MCI_AD_difference = []
    # print tadpole_dp.head()
    rids = tadpole_dp['RID'].values.tolist()
    rids = set(rids)

    for id in rids:
        data[id] = tadpole_dp.loc[tadpole_dp['RID'] == id]

    for key, value in data.iteritems():
        prev_obj = None
        for index, row in value.iterrows():
            date = row['EXAMDATE']
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            if prev_obj is None and row['DXCHANGE'] == 2:
                prev_obj = [date_obj, value.loc[value['EXAMDATE'] == date], 2]
            elif prev_obj is not None and (row['DXCHANGE'] == 3 or row['DXCHANGE'] == 5):
                difference = date_obj - prev_obj[0]
                #print date_obj, prev_obj[0]

                #print "Seconds:", difference.total_seconds()
                #print "Days", ((difference.total_seconds() /60)/60)/24
                # print "Years:", difference.years
                # print "Months:", difference.months
                # print "Days:", difference.days
                MCI_AD_data.append(prev_obj[1])
                MCI_AD_difference.append(abs(((((difference.total_seconds() /60)/60)/24)/365)*12))
                break
    # print "len MCI_AD:", len(MCI_AD_data)
    MCI_AD_data = pd.concat(MCI_AD_data)

    MCI_AD_data['DXCHANGE'].replace([2], [5], inplace=True)

    final_data = MCI_AD_data.copy()

    months_to_AD = [Months_to_Classification(m) for m in MCI_AD_difference]

    final_data['MONTHSAD'] = np.asarray(months_to_AD)

    final_data = final_data.sample(frac=1)
    # print final_data.head()
    # print pd.DataFrame(months_to_AD).describe()
    # print 'final data shape:', final_data.shape
    # print 'final data len:', len(final_data)
    # print ('Total data shape:', tadpole_dp.shape)

    final_data['DX_bl'].replace(['LMCI', 'EMCI', 'SMC'], ['MCI', 'MCI', 'MCI'], inplace=True)
    # print "Unique:", final_data.DXCHANGE.unique

    train_index = int(len(final_data) * 0.8)

    d1_data = final_data[:train_index]
    d2_data = final_data[train_index:]

    d1_data.to_csv(trainfile, encoding='utf-8', index=False)
    d2_data.to_csv(testfile, encoding='utf-8', index=False)

    # final_data.to_csv(trainfile, encoding='utf-8', index=False)

def Months_to_Classification(months):
    if months < 12:
        return 0
    elif months>=12 and months<24:
        return 1
    elif months>=24 and months<36:
        return 2
    else:
        return 3

def ProcessForecast(indata=context + r"/trunk/src/train/TADPOLE_train_time.csv"):

    user_longitudinal_data = {}
    c_data = tadpole_dp = pd.read_csv(indata, low_memory=False)
    # print c_data.head()

    print "head", c_data.shape
    for i in range(-1, 10):
        print i, c_data.loc[tadpole_dp['MONTHSAD'] == i].shape

    print c_data.loc[tadpole_dp['MONTHSAD'] == np.nan].shape

    print c_data.RID.unique().shape

    for index, row in c_data.iterrows():
        if row['RID'] in user_longitudinal_data:
            user_longitudinal_data[row['RID']].append(row)
        else:
            user_longitudinal_data[row['RID']] = [row]

     # print len(user_longitudinal_data)

    '''
    for key, value in user_longitudinal_data.iteritems():
        #print "Key:**", key
        for val in value:
            if val['DXCHANGE'] == 4:
                print val['DX_bl'], ":", val['DXCHANGE'], ':', val['EXAMDATE']
        # print
    '''

if __name__ == "__main__":

    SplitTrainTestDataTime()
    # ProcessForecast()

    # train_cn_data, train_ad_data, train_mci_data = SplitDXData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv")
    # test_cn_data, test_ad_data, test_mci_data = SplitDXData(indata=context + r"/trunk/src/test/TADPOLE_D2.csv")

    # print ('Train CN data shape:', train_cn_data.shape)
    # print ('Train AD data shape:', train_ad_data.shape)
    # print ('Test CN data shape:', test_cn_data.shape)
    # print ('Test AD data shape:', test_ad_data.shape)
