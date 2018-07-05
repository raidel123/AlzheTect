import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    context = os.getcwd().split('\\')
else:
    # Ubuntu
    context = os.getcwd().split('/')

context = '/'.join(context[:context.index('AlzheTect') + 1])

sys.path.append(context + "/trunk/src/utils")

import splitdata as sd
import dnnutils as dnnu
import plotresults as pr

def svm_train(src=r"../train/TADPOLE_train.csv"):
    model_data = GetModelDataCSV(src)
    predict_data = GetModelDataCSV(r"../test/TADPOLE_test.csv")
    predict_data = np.array(predict_data.drop(['DX_bl'], 1))
    lb = LabelBinarizer()

    X = np.array(model_data.drop(['DX_bl'], 1))
    Y = np.array(model_data['DX_bl'])
    Y = lb.fit_transform(Y)

    X = preprocessing.scale(X)

    print X, Y
    print (len(X), len(Y))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
    # X_pred, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()    # n_jobs=-1)
    # clf = svm.SVR()
    clf.fit(X_train, Y_train)

    test_accuracy = clf.score(X_test, Y_test)
    test_accuracy2 = clf.predict(X_test)
    test_accuracy2 = [ResulUnbinarizer(a) for a in test_accuracy2]
    predict_accuracy = clf.predict(predict_data)
    print test_accuracy
    # print "**Predict Accuracy**", predict_accuracy

    print "len X", test_accuracy2
    print "len Y", Y_test.shape

    print len(X_test[0])
    print X_test[:,0]

    # Plot outputs
    plt.scatter(X_test[:,0], Y_test,  color='black')
    plt.plot(predict_data, predict_accuracy, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.savefig('foo.png')


def TransformData(data):
    # TODO: remove print
    # print ('Data train data shape:', data.shape)
    # data = data.drop(columns=['DX_bl'])
    data = data.drop(columns=['DX'])
    data = data.drop(columns=['RID'])
    data = data.drop(columns=['EXAMDATE'])
    transformed = GenderToInt(data)
    transformed = transformed.convert_objects(convert_numeric=True)
    transformed = nan_padding(transformed)

    # print transformed.dtypes

    return transformed

def ResulUnbinarizer(val):
    if val == 0:
        return "AD"
    else:
        return "CN"

# if any of the fields in the top line are changed in the list below
# change the value within nan_padding() function for the relevant fields
# starting positions
def GetRelevantFields():
    return  [
            'RID','DX_bl','DX','EXAMDATE',
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
    for column in GetRelevantFields()[4:]:
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
    split_classes = sd.SplitClassData(indata=model_dp, file=False)
    return TransformData(split_classes)

def GetClass(data):
    fields = []
    for val in data:
        if val == "AD":
            fields.append(np.array([np.int64(0)]))
        elif val == 'CN':
            fields.append(np.array([np.int64(1)]))
        else:
            fields.append(np.array([np.int64(2)]))

    fields = np.array(fields)
    return fields

def get_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["DX_bl"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)
    #data_y = GetClass(data_y)
    print 'type:', type(data_y), data_y
    print 'innertype:', type(data_y[0]), data_y[0]
    print 'innerinnertype:', type(data_y[0][0]), data_y[0][0]
    # print 'type', type(data_y)

    data_x = data.drop(["DX_bl"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y



if __name__ == "__main__":
    svm_train()
