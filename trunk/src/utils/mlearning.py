import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
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

# ------------------------ K-Nearest Neighbors Classifier ---------------------------

def knn_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/knn/knnmodel2.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = sd.SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    print X, Y
    print (len(X), len(Y))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    # clf = svm.SVR()
    clf.fit(X_train, Y_train)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "test score:", test_score
    print "**test predict**", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')

    knn_predict()

def knn_predict(model_loc='../trained_model/knn/knnmodel2.pickle'):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(r"../test/TADPOLE_test.csv")
    # return model_dp

    predict_csv = sd.SplitClassData(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DX_bl'], 1))

    predict_data = preprocessing.scale(predict_data)

    prediction = clf.predict(predict_data)
    print "**Predict Accuracy**", prediction
    probability = clf.predict_proba(predict_data)
    print "**probability**", probability


    results = predict_csv[['RID', 'DX_bl']].copy()
    results['results'] = [ResulUnbinarizer(pred) for pred in prediction]
    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    print results

    results.to_csv(context + r"/trunk/results/knnresults.csv",index=False)

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

# ------------------------ SVM Classifier ---------------------------

def svm_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/svm/svmmodel2.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = sd.SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    print X, Y
    print (len(X), len(Y))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = svm.SVC(probability=True)
    clf.fit(X_train, Y_train)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "test score:", test_score
    print "**test predict**", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')

    svm_predict()

def svm_predict(model_loc='../trained_model/svm/svmmodel2.pickle'):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(r"../test/TADPOLE_test.csv")
    # return model_dp

    predict_csv = sd.SplitClassData(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DX_bl'], 1))

    predict_data = preprocessing.scale(predict_data)

    prediction = clf.predict(predict_data)
    probability = clf.predict_proba(predict_data)
    print "**Prediction**", prediction
    print "**probability**", probability

    results = predict_csv[['RID', 'DX_bl']].copy()
    results['results'] = [ResulUnbinarizer(pred) for pred in prediction]
    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    print results

    results.to_csv(context + r"/trunk/results/svmresults.csv",index=False)

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

# ------------------------ Helper Functions ---------------------------

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
    elif val == 1:
        return "CN"
    else:
        return "MCI"

def Resulbinarizer(val):
    if val == 'AD':
        return 0
    elif val == 'CN':
        return 1
    else:
        return 2

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

    return model_dp

def SplitClassData(indata=context + r"/trunk/src/train/TADPOLE_D1.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    cn_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "CN"]
    ad_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "AD"]
    mci_data = tadpole_dp.loc[tadpole_dp['DX_bl'] == "MCI"]
    # print mci_data.head()
    c_data = pd.concat([ad_data,cn_data, mci_data])
    # c_data = pd.concat([ad_data,cn_data])

    print ('Total data shape:', tadpole_dp.shape)
    return c_data
    # return tadpole_dp

if __name__ == "__main__":
    knn_train()
    knn_predict()

    svm_train()
    svm_predict()
