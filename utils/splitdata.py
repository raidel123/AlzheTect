import tensorflow as tf
import numpy as np
import pandas as pd


def SplitTrainTestData(indata=r"../tadpole/TADPOLE_D1_D2.csv",trainfile=r"../tadpole/TADPOLE_D1.csv",testfile=r"../tadpole/TADPOLE_D2.csv"):
    # input data from tadpole mixed file and separate into training and testing

    tadpole_dp = pd.read_csv(indata, low_memory=False)

    train_data = tadpole_dp.loc[tadpole_dp['D1'] == 1]
    test_data = tadpole_dp.loc[tadpole_dp['D2'] == 1]

    train_data.to_csv(trainfile, encoding='utf-8', index=False)
    test_data.to_csv(testfile, encoding='utf-8', index=False)

    print ('Total data shape:', tadpole_dp.shape)
    print ('Train data shape:', train_data.shape)
    print ('Test data shape:', test_data.shape)

def SplitDXData(indata=r"../train/TADPOLE_D1.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    cn_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "CN"]
    ad_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "AD"]

    print ('Total data shape:', tadpole_dp.shape)
    return cn_data, ad_data

def SplitDXDataCSV(indata=r"../train/TADPOLE_D1.csv", cnfile=r"../train/TADPOLE_D1_CN.csv", adfile=r"../train/TADPOLE_D1_AD.csv"):
    cn_data, ad_data = SplitDXData(indata=indata);

    cn_data.to_csv(cnfile, encoding='utf-8', index=False)
    ad_data.to_csv(adfile, encoding='utf-8', index=False)

    print ('CN data shape:', cn_data.shape)
    print ('AD data shape:', ad_data.shape)

if __name__ == "__main__":
    cn_data, ad_data = SplitDXData(indata=r"../test/TADPOLE_D2.csv")

    print ('CN data shape:', cn_data.shape)
    print ('AD data shape:', ad_data.shape)
