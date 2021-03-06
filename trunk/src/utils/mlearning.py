import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
import warnings
import tensorflow as tf
# import plotly.plotly as py
# py.sign_in('raidel123', 'bASu5tBeMP3WTX69vm9S')
# import plotly.graph_objs as go
import seaborn as sns
import xgboost as xgb

warnings.filterwarnings("ignore")

# Build Neural Network
from collections import namedtuple

# Feature Engineering
from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import style
# style.use("ggplot")

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict

'''
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
'''
# ------------------------ K-Nearest Neighbors Classifier ---------------------------

def knn_train(src=r"../train/TADPOLE_train_MCI.csv", model_loc='../trained_model/knn/knnmodel4.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassDataCN(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DXCHANGE'], 1))
    Y = np.array(tdata['DXCHANGE'])
    Y = np.array([ResulbinarizerCN(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=7)

    # kfolds cross validation
    # creating odd list of K for KNN
    neighbors_lst = list(range(1,50,2))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors_lst:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors_lst[MSE.index(min(MSE))]
    print "The optimal number of neighbors is %d" % optimal_k

    '''
    # plot misclassification error vs k
    plt.plot(neighbors_lst, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.savefig('foo.png')
    '''

    clf = neighbors.KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=-1)
    clf.fit(X_train, Y_train)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "KNN test score:", test_score
    print "KNN test predict:", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')
    # plt.plot()

    # knn_predict()

def knn_predict(model_loc='../trained_model/knn/knnmodel4.pickle', input_data="../test/TADPOLE_test_MCI.csv", appcontext=None):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(input_data)
    predict_csv = SplitClassDataCN(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DXCHANGE'], 1))
    predict_lbl = np.array(split_classes['DXCHANGE'])

    predict_data = preprocessing.scale(predict_data)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(predict_lbl)
    encoded_Y = encoder.transform(predict_lbl)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    prediction = clf.predict(predict_data)
    # print "**Predict Accuracy**", prediction
    probability = clf.predict_proba(predict_data)
    # print "**probability**", probability

    results = predict_csv[['RID', 'DXCHANGE']].copy()
    results['results'] = [ResulUnbinarizerCN(pred) for pred in prediction]
    results['MONTHSAD'] = [None] * len(results['results'])

    scores = accuracy_score(results['DXCHANGE'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['DXCHANGE'], results['results'])
    print conf_mat

    '''
    trace = go.Heatmap(z=conf_mat,
                   x=['CN', 'MCI', 'AD', 'MCI-to-AD'],
                   y=['CN', 'MCI', 'AD', 'MCI-to-AD'])
    data=[trace]
    py.plot(data, filename='knn-dx-heatmap')
    '''

    '''
    fig2=plt.figure()
    fig2.add_subplot(111)
    sns.heatmap(conf_mat,annot=True,square=True,cbar=False,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('knn_heatmap_dx.png')
    '''

    list_indexes = []
    for i in range(len(results['results'])):
        if results['results'][i] == 5:
            list_indexes.append(i)
            # print "i:", i

    # print "len:", len(list_indexes)
    # print len(results['results'])
    extra_test_cases = []
    y_pred = []
    for i in list_indexes:
        extra_test_cases.append(predict_csv.values.tolist()[i])

    print "Type:", type(extra_test_cases)
    extra_pd = pd.DataFrame(extra_test_cases, columns=predict_csv.columns.tolist())


    if appcontext is None:
        time_results = keras_test_time(input_data=extra_pd, model_loc='../trained_model/keras/kerasmodel4time.yaml', weights_loc='../trained_model/keras/kerasmodel4time.h5', local=False)
    else:
        time_results = keras_test_time(input_data=extra_pd, model_loc=appcontext+'/src/trained_model/keras/kerasmodel4time.yaml', weights_loc=appcontext+'/src/trained_model/keras/kerasmodel4time.h5', local=False)
    # print results

    index = 0
    time_results_class = time_results['results'].values.tolist()
    for index2 in range(len(results['results'])):
        if results['results'][index2] == 5:
            results['MONTHSAD'][index2] = time_results_class[index]
            index+=1

    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # print results

    # results.to_csv("../../results/uploads/results.csv",index=False)

    K.clear_session()

    return results

# ------------------------ SVM Classifier ---------------------------

def svm_train(src=r"../train/TADPOLE_train_MCI.csv", model_loc='../trained_model/svm/svmmodel4.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassDataCN(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DXCHANGE'], 1))
    Y = np.array(tdata['DXCHANGE'])
    Y = np.array([ResulbinarizerCN(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    '''
    # parameter tuning
    n_splits = 10
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)

    log_cols = ["C", "gamma", "Accuracy"]
    log 	 = pd.DataFrame(columns=log_cols)

    acc_dict = {}

    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]


        for lambd in [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1]:
            for gamma in [1.0E-4, 1.0E-3, 1.0E-2, 1.0E-1, 1.0, 10.0]:

                clf = svm.SVC(probability=True, C=1/lambd, gamma=gamma)
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
                if lambd in acc_dict:
                    if gamma in acc_dict[lambd]:
                        acc_dict[lambd][gamma] += acc
                    else:
                        acc_dict[lambd][gamma] = acc
                else:
                    acc_dict[lambd] = {}
                    acc_dict[lambd][gamma] = acc


    for lambd in acc_dict:
        for gamma in acc_dict[lambd]:
            acc_value = acc_dict[lambd][gamma] / n_splits
            log_entry = pd.DataFrame([[lambd, gamma, acc_value]], columns=log_cols)
            log = log.append(log_entry)

    #print ('Classifier Accuracy')
    #print (log)
    #print ()

    plt.figure()

    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')

    heatmap_data = log.pivot("C", "gamma", "Accuracy")
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f')
    plt.savefig('foo4.png')
    '''

    lambd = 0.001
    gamma = 0.001

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = svm.SVC(probability=True, C=1/lambd, gamma=gamma)
    clf.fit(X_train, Y_train)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "SVM test score:", test_score
    print "SVM test predict:", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')

    # svm_predict()


def svm_predict(model_loc='../trained_model/svm/svmmodel4.pickle', input_data="../test/TADPOLE_test_MCI.csv", appcontext=None):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(input_data)
    # return model_dp

    predict_csv = SplitClassDataCN(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DXCHANGE'], 1))

    predict_data = preprocessing.scale(predict_data)

    prediction = clf.predict(predict_data)
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", prediction
    # print "**probability**", probability

    results = predict_csv[['RID', 'DXCHANGE']].copy()
    results['results'] = [ResulUnbinarizerCN(pred) for pred in prediction]
    results['MONTHSAD'] = [None] * len(results['results'])

    scores = accuracy_score(results['DXCHANGE'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['DXCHANGE'], results['results'])
    print conf_mat

    '''
    fig2=plt.figure()
    fig2.add_subplot(111)
    sns.heatmap(conf_mat,annot=True,square=True,cbar=False,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('svm-dx-heatmap.png')
    '''

    list_indexes = []
    for i in range(len(results['results'])):
        if results['results'][i] == 5:
            list_indexes.append(i)
            # print "i:", i

    # print "len:", len(list_indexes)
    # print len(results['results'])
    extra_test_cases = []
    y_pred = []
    for i in list_indexes:
        extra_test_cases.append(predict_csv.values.tolist()[i])

    print "Type:", type(extra_test_cases)
    extra_pd = pd.DataFrame(extra_test_cases, columns=predict_csv.columns.tolist())


    if appcontext is None:
        time_results = keras_test_time(input_data=extra_pd, model_loc='../trained_model/keras/kerasmodel4time.yaml', weights_loc='../trained_model/keras/kerasmodel4time.h5', local=False)
    else:
        time_results = keras_test_time(input_data=extra_pd, model_loc=appcontext+'/src/trained_model/keras/kerasmodel4time.yaml', weights_loc=appcontext+'/src/trained_model/keras/kerasmodel4time.h5', local=False)
    # print results

    index = 0
    time_results_class = time_results['results'].values.tolist()
    for index2 in range(len(results['results'])):
        if results['results'][index2] == 5:
            results['MONTHSAD'][index2] = time_results_class[index]
            index+=1

    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # print results

    # results.to_csv(r"../../results/svmresults.csv",index=False)

    return results

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')
# ------------------------ DNN Classifier ---------------------------

def keras_trainCN(src=r"../train/TADPOLE_train_MCI.csv", model_loc='../trained_model/keras/kerasmodel4CN.yaml', weights_loc="../trained_model/keras/kerasmodel4CN.h5"):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    model_data = GetModelDataCSV(src)
    split_classes = SplitClassDataCN(indata=model_data, file=False)
    tdata = TransformData(split_classes)
    # original_X_cp = pd.DataFrame.copy(tdata)

    X = np.array(tdata.drop(['DXCHANGE'], 1))
    Y = np.array(tdata['DXCHANGE'])
    Y = np.array([ResulbinarizerCN(label) for label in Y])

    X = preprocessing.scale(X)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    clf = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)
    # clf = baseline_model()

    # kfold  = KFold(n_splits=10, shuffle=True, random_state=seed)

    # results = cross_val_score(clf, X, dummy_Y, cv=kfold)
    # print("Result: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, dummy_Y, test_size=0.2)

    clf.fit(X_train, Y_train) # , callbacks=callbacks_list)

    scores = clf.model.evaluate(X_train, Y_train, verbose=0)
    print("%s: %.2f%%" % (clf.model.metrics_names[1], scores[1]*100))

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "keras test_score:", test_score
    print "keras test_predict:", test_predict

    # serialize model to YAML
    model_yaml = clf.model.to_yaml()
    with open(model_loc, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    clf.model.save_weights(weights_loc)
    print("Saved model to disk")

    '''
    # saving model
    model_yaml = clf.model.to_yaml()
    # open(model_loc, 'w').write(json_model)
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # saving weights
    clf.model.save_weights("../trained_model/keras/kerasmodel2.h5")
    '''

def keras_testCN(model_loc='../trained_model/keras/kerasmodel4CN.yaml', weights_loc="../trained_model/keras/kerasmodel4CN.h5", input_data="../test/TADPOLE_test_MCI.csv", appcontext=None):

    predict_csv = GetModelDataCSV(input_data)
    # return model_dp

    predict_csv = SplitClassDataCN(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DXCHANGE'], 1))
    predict_lbl = np.array(split_classes['DXCHANGE'])

    predict_data = preprocessing.scale(predict_data)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(predict_lbl)
    encoded_Y = encoder.transform(predict_lbl)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    # clf = build_by_loading() # KerasClassifier(build_fn=build_by_loading, nb_epoch=10, batch_size=5, verbose=1)

    # load YAML and create model
    yaml_file = open(model_loc, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    clf = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    clf.load_weights(weights_loc)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    # clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = clf.evaluate(predict_data, dummy_Y, verbose=0)
    # print("%s: %.2f%%" % (clf.metrics_names[1], score[1]*100))

    prediction = clf.predict(predict_data)
    # for pred in prediction:
    prediction = [np.argmax(pred) for pred in prediction]
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", pred
    # print "**probability**", probability

    results = predict_csv[['RID', 'DXCHANGE']].copy()
    results['results'] = [ResulUnbinarizerCN(pred) for pred in prediction]
    results['MONTHSAD'] = [None] * len(results['results'])

    scores = accuracy_score(results['DXCHANGE'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['DXCHANGE'], results['results'])
    print conf_mat

    '''
    fig2=plt.figure()
    fig2.add_subplot(111)
    sns.heatmap(conf_mat,annot=True,square=True,cbar=False,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('keras-dx-heatmap.png')
    '''

    list_indexes = []
    for i in range(len(results['results'])):
        if results['results'][i] == 5:
            list_indexes.append(i)
            # print "i:", i

    # print "len:", len(list_indexes)
    # print len(results['results'])
    extra_test_cases = []
    y_pred = []
    for i in list_indexes:
        extra_test_cases.append(predict_csv.values.tolist()[i])

    print "Type:", type(extra_test_cases)
    extra_pd = pd.DataFrame(extra_test_cases, columns=predict_csv.columns.tolist())


    if appcontext is None:
        time_results = keras_test_time(input_data=extra_pd, model_loc='../trained_model/keras/kerasmodel4time.yaml', weights_loc='../trained_model/keras/kerasmodel4time.h5', local=False)
    else:
        time_results = keras_test_time(input_data=extra_pd, model_loc=appcontext+'/src/trained_model/keras/kerasmodel4time.yaml', weights_loc=appcontext+'/src/trained_model/keras/kerasmodel4time.h5', local=False)
    # print results

    index = 0
    time_results_class = time_results['results'].values.tolist()
    for index2 in range(len(results['results'])):
        if results['results'][index2] == 5:
            results['MONTHSAD'][index2] = time_results_class[index]
            index+=1

    # print results
    # print results.values.tolist()
    '''
    scores = accuracy_score(results['DXCHANGE'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['DXCHANGE'], results['results'])
    print conf_mat
    '''

    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # results.to_csv(r"../../results/kerasresults.csv",index=False)

    return results

def keras_train_time(src=r"../train/TADPOLE_train_time.csv", model_loc='../trained_model/keras/kerasmodel4time.yaml', weights_loc="../trained_model/keras/kerasmodel4time.h5"):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    model_data = GetModelDataCSV2(src)
    # split_classes = SplitClassDataTime(indata=model_data, file=False)
    tdata = TransformData2(model_data)
    # original_X_cp = pd.DataFrame.copy(tdata)

    X = np.array(tdata.drop(['MONTHSAD'], 1))
    Y = np.array(tdata['MONTHSAD'])
    Y = np.array([ResulbinarizerTime(label) for label in Y])

    X = preprocessing.scale(X)

    # print X

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    clf = KerasClassifier(build_fn=baseline_model_time, epochs=200, batch_size=4, verbose=0)
    # clf = baseline_model()

    # kfold  = KFold(n_splits=10, shuffle=True, random_state=seed)

    # results = cross_val_score(clf, X, dummy_Y, cv=kfold)
    # print("Result: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, dummy_Y, test_size=0.2)

    # filepath = "../trained_model/keras/kerasmodel2.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]

    clf.fit(X_train, Y_train) # , callbacks=callbacks_list)

    # plot_model(clf, to_file='model.png')

    scores = clf.model.evaluate(X_train, Y_train, verbose=0)
    print("%s: %.2f%%" % (clf.model.metrics_names[1], scores[1]*100))

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "keras test_score:", test_score
    print "keras test_predict:", test_predict

    # serialize model to YAML
    model_yaml = clf.model.to_yaml()
    with open(model_loc, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    clf.model.save_weights(weights_loc)
    print("Saved model to disk")

    '''
    # saving model
    model_yaml = clf.model.to_yaml()
    # open(model_loc, 'w').write(json_model)
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # saving weights
    clf.model.save_weights("../trained_model/keras/kerasmodel2.h5")
    '''

def keras_test_time(model_loc='../trained_model/keras/kerasmodel4time.yaml', weights_loc="../trained_model/keras/kerasmodel4time.h5", input_data="../test/TADPOLE_test_time.csv", local=True):

    if local:
        predict_csv = GetModelDataCSV2(input_data) # com1
        split_classes = TransformData2(predict_csv)   # com1
        split_classes = split_classes.drop(columns=['MONTHSAD'])
        original_pandas = predict_csv
    else:
        split_classes = TransformData2(input_data)   # com1
        original_pandas = input_data

    predict_data = np.array(split_classes) # com1
    # predict_lbl = np.array(pred_y)    # split_classes['MONTHSAD']) # com1
    predict_data = preprocessing.scale(predict_data)

    '''
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(predict_lbl)
    encoded_Y = encoder.transform(predict_lbl)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)
    '''
    # clf = build_by_loading() # KerasClassifier(build_fn=build_by_loading, nb_epoch=10, batch_size=5, verbose=1)

    # load YAML and create model
    yaml_file = open(model_loc, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    clf = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    clf.load_weights(weights_loc)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = clf.evaluate(predict_data, dummy_Y, verbose=0)
    # print("%s: %.2f%%" % (clf.metrics_names[1], score[1]*100))

    prediction = clf.predict(predict_data)
    # for pred in prediction:
    prediction = [np.argmax(pred) for pred in prediction]
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", pred
    # print "**probability**", probability

    if local:
        results = original_pandas[['RID', 'MONTHSAD']].copy()
        results['results'] = [ResulUnbinarizerTime(pred) for pred in prediction]


        scores = accuracy_score(results['MONTHSAD'], results['results'])
        print scores

        conf_mat = confusion_matrix(results['MONTHSAD'], results['results'])
        print conf_mat

    else:
        results = original_pandas[['RID']].copy()
        results['results'] = [ResulUnbinarizerTime(pred) for pred in prediction]

    '''
    scores = accuracy_score(results['MONTHSAD'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['MONTHSAD'], results['results'])
    print conf_mat
    '''

    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # results.to_csv(r"../../results/kerastimeresults.csv",index=False)

    print results

    return results

def keras_test_time2(model_loc='../trained_model/keras/kerasmodel4time.yaml', weights_loc="../trained_model/keras/kerasmodel4time.h5", input_data="../test/TADPOLE_test_time.csv"):

    predict_csv = GetModelDataCSV2(input_data) # com1
    split_classes = TransformData2(predict_csv)   # com1
    # split_classes = split_classes.drop(columns=['MONTHSAD'])
    # pred_y = split_classes['MONTHSAD']

    original_pandas = predict_csv

    predict_data = np.array(split_classes.drop(columns=['MONTHSAD'])) # com1
    predict_lbl = np.array(split_classes['MONTHSAD'])    # split_classes['MONTHSAD']) # com1

    predict_data = preprocessing.scale(predict_data)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(predict_lbl)
    encoded_Y = encoder.transform(predict_lbl)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    # clf = build_by_loading() # KerasClassifier(build_fn=build_by_loading, nb_epoch=10, batch_size=5, verbose=1)

    # load YAML and create model
    yaml_file = open(model_loc, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    clf = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    clf.load_weights(weights_loc)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = clf.evaluate(predict_data, dummy_Y, verbose=0)
    print("%s: %.2f%%" % (clf.metrics_names[1], score[1]*100))

    prediction = clf.predict(predict_data)
    # for pred in prediction:
    prediction = [np.argmax(pred) for pred in prediction]
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", pred
    # print "**probability**", probability

    results = original_pandas[['RID', 'MONTHSAD']].copy()
    results['results'] = [ResulUnbinarizerTime(pred) for pred in prediction]

    scores = accuracy_score(results['MONTHSAD'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['MONTHSAD'], results['results'])
    print conf_mat

    fig2=plt.figure()
    fig2.add_subplot(111)
    sns.heatmap(conf_mat,annot=True,square=True,cbar=False,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('keras-time-heatmap.png')

    '''
    scores = accuracy_score(results['MONTHSAD'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['MONTHSAD'], results['results'])
    print conf_mat
    '''

    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # results.to_csv(r"../../results/kerastimeresults.csv",index=False)

    print results

    return results


# ----------------------SVM predict time ---------------------------------
def svm_train_time(src=r"../train/TADPOLE_train_time.csv", model_loc='../trained_model/svm/svmmodel4time.pickle'):
    model_data = GetModelDataCSV2(src)
    # split_classes = SplitClassDataCN(indata=model_data, file=False)
    tdata = TransformData2(model_data)

    X = np.array(tdata.drop(['MONTHSAD'], 1))
    Y = np.array(tdata['MONTHSAD'])
    Y = np.array([ResulbinarizerTime(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    '''
    # parameter tuning
    n_splits = 10
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)

    log_cols = ["C", "gamma", "Accuracy"]
    log 	 = pd.DataFrame(columns=log_cols)

    acc_dict = {}

    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]


        for lambd in [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1]:
            for gamma in [1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2, 1.0E-1]:

                clf = svm.SVC(probability=True, C=1/lambd, gamma=gamma)
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
                if lambd in acc_dict:
                    if gamma in acc_dict[lambd]:
                        acc_dict[lambd][gamma] += acc
                    else:
                        acc_dict[lambd][gamma] = acc
                else:
                    acc_dict[lambd] = {}
                    acc_dict[lambd][gamma] = acc


    for lambd in acc_dict:
        for gamma in acc_dict[lambd]:
            acc_value = acc_dict[lambd][gamma] / n_splits
            log_entry = pd.DataFrame([[lambd, gamma, acc_value]], columns=log_cols)
            log = log.append(log_entry)

    #print ('Classifier Accuracy')
    #print (log)
    #print ()

    plt.figure()

    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')

    heatmap_data = log.pivot("C", "gamma", "Accuracy")
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f')
    plt.savefig('time.png')
    '''

    lambd = 0.03
    gamma = 0.001

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = svm.SVC(probability=True, C=1/lambd, gamma=gamma)
    clf.fit(X_train, Y_train)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    test_score = clf.score(X_test, Y_test)
    test_predict = clf.predict(X_test)

    print "SVM test score:", test_score
    print "SVM test predict:", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')

    # svm_predict()


def svm_predict_time(model_loc='../trained_model/svm/svmmodel4time.pickle', input_data="../test/TADPOLE_test_time.csv"):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV2(input_data)
    # return model_dp

    # predict_csv = SplitClassDataCN(indata=predict_csv, file=False)
    split_classes = TransformData2(predict_csv)

    predict_data = np.array(split_classes.drop(['MONTHSAD'], 1))
    predict_lbl = np.array(split_classes['MONTHSAD'])    # split_classes['MONTHSAD']) # com1

    predict_data = preprocessing.scale(predict_data)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(predict_lbl)
    encoded_Y = encoder.transform(predict_lbl)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    prediction = clf.predict(predict_data)
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", prediction
    # print "**probability**", probability

    results = predict_csv[['RID', 'MONTHSAD']].copy()
    results['results'] = [ResulUnbinarizerTime(pred) for pred in prediction]
    # results['MONTHSAD'] = [None] * len(results['results'])

    scores = accuracy_score(results['MONTHSAD'], results['results'])
    print scores

    conf_mat = confusion_matrix(results['MONTHSAD'], results['results'])
    print conf_mat

    '''
    fig2=plt.figure()
    fig2.add_subplot(111)
    sns.heatmap(conf_mat,annot=True,square=True,cbar=False,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('svm-dx-heatmap.png')
    '''

    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # print results

    # results.to_csv(r"../../results/svmresults.csv",index=False)

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

    return results


# ---------------------- Models and utilities ---------------------------------
def build_by_loading():
    model = load_model('../trained_model/keras/kerasmodel2.h5')
    return model

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define baseline model
def baseline_model_time():
    # create model
    model = Sequential()
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def random_forest_regressor(src=r"../train/TADPOLE_train_MCI.csv"):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassDataCN(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DXCHANGE'], 1))
    Y = np.array(tdata['DXCHANGE'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    names = list(tdata.drop(['DXCHANGE'], 1).columns.values)

    rf = RandomForestClassifier(n_estimators=500, max_features=20, n_jobs=-1, verbose=1)
    scores = defaultdict(list)

    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = accuracy_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = accuracy_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print "Features sorted by their score:"
    for item in sorted([(round(np.mean(score), 8), feat) for feat, score in scores.items()], reverse=True)[:20]:
        print item

    json.dump(scores, open('../trained_model/random_forest/rfr_scores2.json', 'w'))

def rfc_results():
    dict_load = json.load(open('../trained_model/random_forest/rfr_scores2.json', 'r'))
    for item in sorted([(round(np.mean(dict_load), 8), feat) for feat, dict_load in dict_load.items()], reverse=True)[:20]:
        print item

# ------------------------ Helper Functions ---------------------------

def TransformData(data):
    # TODO: remove print
    # print ('Data train data shape:', data.shape)
    data = data.drop(columns=['DX_bl'])
    data = data.drop(columns=['DX'])
    data = data.drop(columns=['RID'])
    data = data.drop(columns=['EXAMDATE'])
    if 'PTGENDER' in data:
        data = data.drop(columns=['PTGENDER'])
    if 'MONTHSAD' in data:
        data = data.drop(columns=['MONTHSAD'])
    # transformed = GenderToInt(data)
    transformed = data.convert_objects(convert_numeric=True)
    transformed = nan_padding(transformed)
    # transformed = transformed.fillna(transformed.mean())

    # print transformed.dtypes

    return transformed

def TransformData2(data):
    # TODO: remove print
    # print ('Data train data shape:', data.shape)
    data = data.drop(columns=['DX_bl'])
    data = data.drop(columns=['DX'])
    data = data.drop(columns=['RID'])
    data = data.drop(columns=['EXAMDATE'])
    data = data.drop(columns=['DXCHANGE'])
    if 'PTGENDER' in data:
        data = data.drop(columns=['PTGENDER'])
    # transformed = GenderToInt(data)
    transformed = data.convert_objects(convert_numeric=True)
    transformed = nan_padding(transformed)
    # transformed = transformed.fillna(transformed.mean())

    # print transformed.dtypes
    return transformed

def ResulUnbinarizer(val):
    if val == 0:
        return 1
    elif val == 1:
        return 2
    else:
        return 3

def Resulbinarizer(val):
    if val == 1:
        return 0
    elif val == 4:
        return 1
    else:
        return 2

def ResulUnbinarizerCN(val):
    if val == 0:
        return 1
    elif val == 1:
        return 2
    elif val == 2:
        return 3
    else:
        return 5

def ResulbinarizerCN(val):
    if val == 1:
        return 0
    elif val == 2:
        return 1
    elif val == 3:
        return 2
    else:
        return 3

def ResulUnbinarizerTime(val):
    if val == 0:
        return 0
    elif val == 1:
        return 1
    elif val == 2:
        return 2
    else:
        return 3

def ResulbinarizerTime(val):
    if val == 0:
        return 0
    elif val == 1:
        return 1
    elif val == 2:
        return 2
    else:
        return 3

# if any of the fields in the top line are changed in the list below
# change the value within nan_padding() function for the relevant fields
# starting positions
def GetSetRelevantFields():
    return  [
            'RID','DX_bl','DX','EXAMDATE','DXCHANGE',
            'MMSE_bl',
            'CDRSB',
            'ADAS13',
            'ADAS11',
            'RAVLT_immediate',
            'MMSE',
            'APOE4',
            'LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16',
            'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'Hippocampus',
            'ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST39TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST83CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'AV45',
            'WholeBrain',
            'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
            ]

def GetSetRelevantFieldsTime():
    return  [
            'RID','DX_bl','DX','EXAMDATE','DXCHANGE', 'MONTHSAD',
            'MMSE_bl',
            'CDRSB',
            'ADAS13',
            'ADAS11',
            'RAVLT_immediate',
            'MMSE',
            'APOE4',
            'LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16',
            'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'Hippocampus',
            'ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST39TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST83CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'AV45',
            'WholeBrain',
            'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
            ]

def GetAllRelevantFields():
    return  [
            'RID','DX_bl','DX','EXAMDATE', 'DXCHANGE',
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
            'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16',
            'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16','LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16',
            'ST89SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST130TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST130TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
            'ST12SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16','LEFT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16',
            'ST40TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST40TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST117SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16','ST109TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST109TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            'ST110TA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16','ST110TS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
            ]

# getting cannot convert string to float errors
def nan_padding(data):
    for column in data.columns:
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

    model_dp = file_dp[GetSetRelevantFields()]

    return model_dp

def GetModelDataCSV2(indata):
    # load data (low memory: supress memory warning)
    file_dp = pd.read_csv(indata, low_memory=False)

    model_dp = file_dp[GetSetRelevantFieldsTime()]

    return model_dp

def SplitClassData(indata=r"/trunk/src/train/TADPOLE_train.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    cn_data = tadpole_dp.loc[tadpole_dp['DXCHANGE'] == 1]
    mci_data = tadpole_dp.loc[tadpole_dp['DXCHANGE'] == 2]
    ad_data = tadpole_dp.loc[tadpole_dp['DXCHANGE'] == 3]
    # print mci_data.head()
    c_data = pd.concat([ad_data,cn_data, mci_data])
    #c_data = pd.concat([ad_data,cn_data])

    print ('Total data shape:', tadpole_dp.shape)
    return c_data
    # return tadpole_dp

def SplitClassDataMCI(indata=r"/trunk/src/train/TADPOLE_train.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    mci_mci_data = tadpole_dp.loc[tadpole_dp['DXCHANGE'] == 2]
    mci_ad_data = tadpole_dp.loc[tadpole_dp['DXCHANGE'] == 5]

    # print mci_data.head()
    c_data = pd.concat([ad_data,cn_data, mci_data])
    #c_data = pd.concat([ad_data,cn_data])

    print ('Total data shape:', tadpole_dp.shape)
    return c_data
    # return tadpole_dp

def SplitClassDataCN(indata=r"/trunk/src/train/TADPOLE_train.csv", file=True):

    if file:
        tadpole_dp = pd.read_csv(indata, low_memory=False)
    else:
        tadpole_dp = indata

    return tadpole_dp

if __name__ == "__main__":
    # knn_train()
    # knn_predict()

    # svm_train()
    # svm_predict()

    # kmeans_train()
    # kmeans_predict()

    # mean_shift_train()

    # TrainModel()
    # TestModel()

    # keras_trainCN()
    # keras_testCN()

    # keras_train_time()
    # keras_test_time()
    # keras_test_time2()

    # svm_train_time()
    # svm_predict_time()

    random_forest_regressor()
    rfc_results()
