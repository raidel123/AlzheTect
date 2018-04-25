#!/usr/bin/env python

import pandas as pd
import splitdata as sd

# Feature Engineering
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# if any of the fields in the top line are changed in the list below
# change the value within nan_padding() function for the relevant fields
# starting positions
def GetRelevantFields():
    return  [
            'RID','DX_bl','DX',
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
            ]

# getting cannot convert string to float errors
def nan_padding(data):
    for column in GetRelevantFields()[3:]:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

def GenderToInt(data):
    le = LabelEncoder()
    le.fit(["Male","Female"])
    data["PTGENDER"]=le.transform(data["PTGENDER"])
    return data

def GetModelDataCSV(indata):
    # load data (low memory: supress memory warning)
    file_dp = pd.read_csv(indata, low_memory=False)

    model_dp = file_dp[GetRelevantFields()]

    # return model_dp
    return sd.SplitClassData(indata=model_dp, file=False)

def get_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["DX_bl"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["DX_bl"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y
