# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
import warnings
import tensorflow as tf
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
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

def knn_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/knn/knnmodel2.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    # clf = svm.SVR()
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

def knn_predict(model_loc='../trained_model/knn/knnmodel2.pickle', input_data="../../results/uploads/upload.csv", output_file="../../results/uploads/results.csv", file=True):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(input_data)
    # return model_dp

    predict_csv = SplitClassData(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DX_bl'], 1))

    predict_data = preprocessing.scale(predict_data)

    prediction = clf.predict(predict_data)
    # print "**Predict Accuracy**", prediction
    probability = clf.predict_proba(predict_data)
    # print "**probability**", probability


    results = predict_csv[['RID', 'DX_bl']].copy()
    results['results'] = [ResulUnbinarizer(pred) for pred in prediction]
    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # print results

    # results.to_csv(output_file,index=False)

    return results

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

# ------------------------ SVM Classifier ---------------------------

def svm_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/svm/svmmodel2.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = svm.SVC(probability=True)
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

def svm_predict(model_loc='../trained_model/svm/svmmodel2.pickle', input_data="../../results/uploads/upload.csv"):

    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(input_data)
    # return model_dp

    predict_csv = SplitClassData(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DX_bl'], 1))

    predict_data = preprocessing.scale(predict_data)

    prediction = clf.predict(predict_data)
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", prediction
    # print "**probability**", probability

    results = predict_csv[['RID', 'DX_bl']].copy()
    results['results'] = [ResulUnbinarizer(pred) for pred in prediction]
    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # print results

    # results.to_csv(r"/trunk/results/svmresults.csv",index=False)

    return results

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

# ------------------------ KMeans Classifier ---------------------------

def kmeans_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/kmeans/kmeansmodel2.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    correct = 0
    for i in range(len(X)):
        predict_X = np.array(X[i].astype(float))
        predict_X = predict_X.reshape(-1, len(predict_X))
        pred = clf.predict(predict_X)

        # print pred[0], Y[i]
        if pred[0] == Y[i]:
            correct += 1

    # print correct
    print "kmeans correct percent:***", float(correct)/float(len(X))

    # labels = clf.labels_
    # cluster_centers = clf.cluster_centers_

    '''
    colors = 10*['r','g','b','c','k','y','m']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print X
    for i in range(len(X)):
        ax.scatter(X[i][0], X[i][1], c=colors[labels[i]], marker='o')

    ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker="x",color='k', s=150, linewidths = 5, zorder=10)

    plt.savefig('kmeans.png')

    #test_score = clf.score(X_test, Y_test)
    #test_predict = clf.predict(X_test)

    #print "test score:", test_score
    #print "**test predict**", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')

    # svm_predict()
    '''

def kmeans_predict(model_loc='../trained_model/kmeans/kmeansmodel2.pickle'):
    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(r"../test/TADPOLE_test.csv")
    # return model_dp

    predict_csv = SplitClassData(indata=predict_csv, file=False)
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

    # results.to_csv(r"/trunk/results/svmresults.csv",index=False)

    #colors = 10*['r','g','b','c','k','y','m']
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    #print X
    #for i in range(len(X)):
        #ax.scatter(X[i][0], X[i][1], c=colors[labels[i]], marker='o')

    # ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker="x",color='k', s=150, linewidths = 5, zorder=10)

    # plt.savefig('meanshift.png')

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

# ------------------------ Mean Shift Classifier ---------------------------

def mean_shift_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/means_shift/means_shiftmodel2.pickle'):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)
    original_X_cp = pd.DataFrame.copy(tdata)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    # print X, Y
    # print (len(X), len(Y))

    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    # X, _ = make_blobs(n_samples = 100, cluster_std = 1.5)
    clf = MeanShift()
    clf.fit(X)

    with open(model_loc, 'wb') as f:
        pickle.dump(clf, f)

    labels = clf.labels_
    cluster_centers = clf.cluster_centers_

    print "labels:", labels
    print "cluster_centers:", cluster_centers

    original_X_cp['clusters'] = np.nan
    for i in range(len(X)):
        original_X_cp['clusters'].iloc[i] = labels[i]
    print original_X_cp

    n_clusters_ = len(np.unique(labels))
    ad_rates = {}
    for i in range(n_clusters_):
        temp_df = original_X_cp[(original_X_cp['clusters'] == float(i))]
        # print temp_df.head()

        ad_cluster = temp_df[(temp_df['DX_bl'] == 'AD')]
        # print "ad_cluster", ad_cluster
        ad_rate = float(len(ad_cluster))/float(len(temp_df))
        print "i, rate:", i,ad_rate
        ad_rates[i] = ad_rate

    print "ad_Rates:", ad_rates
    print "n clust:", n_clusters_

    print original_X_cp[(original_X_cp['clusters'] == 0)].describe()

    '''
    correct = 0
    for i in range(len(X)):
        predict_X = np.array(X[i].astype(float))
        predict_X = predict_X.reshape(-1, len(predict_X))
        pred = clf.predict(predict_X)

        # print pred[0], Y[i]
        if pred[0] == Y[i]:
            correct += 1

    # print correct
    print "***correct percent:***", float(correct)/float(len(X))
    '''
    # print(cluster_centers)
    # n_clusters_ = len(np.unique(labels))
    # print("Number of estimated clusters:", n_clusters_)

    #colors = 10*['r','g','b','c','k','y','m']
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    #print X
    #for i in range(len(X)):
        #ax.scatter(X[i][0], X[i][1], c=colors[labels[i]], marker='o')

    # ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker="x",color='k', s=150, linewidths = 5, zorder=10)

    # plt.savefig('meanshift.png')

    #test_score = clf.score(X_test, Y_test)
    #test_predict = clf.predict(X_test)

    #print "test score:", test_score
    #print "**test predict**", test_predict

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_predict, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo.png')

    # svm_predict()

def mean_shift_predict(model_loc='../trained_model/means_shift/means_shiftmodel2.pickle'):
    trained_classifier = open(model_loc ,'rb')
    clf = pickle.load(trained_classifier)

    predict_csv = GetModelDataCSV(r"../test/TADPOLE_test.csv")
    # return model_dp

    predict_csv = SplitClassData(indata=predict_csv, file=False)
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

    results.to_csv(r"/trunk/results/svmresults.csv",index=False)

    # Plot outputs
    # plt.scatter(X_test[:,0], Y_test,  color='black')
    # plt.plot(X_test, test_accuracy2, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.savefig('foo2.png')

# ------------------------ DNN Classifier ---------------------------

def keras_train(src=r"../train/TADPOLE_train.csv", model_loc='../trained_model/keras/kerasmodel2.yaml', weights_loc="../trained_model/keras/kerasmodel2.h5"):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    model_data = GetModelDataCSV(src)
    split_classes = SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)
    # original_X_cp = pd.DataFrame.copy(tdata)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    clf = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    # clf = baseline_model()

    # kfold  = KFold(n_splits=10, shuffle=True, random_state=seed)

    # results = cross_val_score(clf, X, dummy_Y, cv=kfold)
    # print("Result: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, dummy_Y, test_size=0.2)

    # filepath = "../trained_model/keras/kerasmodel2.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]

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



def keras_test(model_loc='../trained_model/keras/kerasmodel2.yaml', weights_loc="../trained_model/keras/kerasmodel2.h5", input_data="../../results/uploads/upload.csv"):

    predict_csv = GetModelDataCSV(input_data)
    # return model_dp

    predict_csv = SplitClassData(indata=predict_csv, file=False)
    split_classes = TransformData(predict_csv)

    predict_data = np.array(split_classes.drop(['DX_bl'], 1))
    predict_lbl = np.array(split_classes['DX_bl'])

    predict_data = preprocessing.scale(predict_data)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(predict_lbl)
    encoded_Y = encoder.transform(predict_lbl)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_Y = np_utils.to_categorical(encoded_Y)

    # clf = build_by_loading() # KerasClassifier(build_fn=build_by_loading, nb_epoch=10, batch_size=5, verbose=1)

    '''
    # loading model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    # clf = model_from_json(open(model_loc).read())

    clf = model_from_yaml(loaded_model_yaml)

    clf.load_weights("../trained_model/keras/kerasmodel2.h5")
    clf.compile(loss='categorical_crossentropy', optimizer='adam')
    '''

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

    # clf = load_model("../trained_model/keras/kerasmodel2.h5")

    # clf = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)
    # load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # clf.model = model_from_json(loaded_model_json)
    # load weights into new model
    # clf.model.load_weights("model.h5")
    # print("Loaded model from disk")

    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    prediction = clf.predict(predict_data)
    # for pred in prediction:
    prediction = [0 if pred[0] > pred[1] else 1 for pred in prediction]
    probability = clf.predict_proba(predict_data)
    # print "**Prediction**", pred
    # print "**probability**", probability

    results = predict_csv[['RID', 'DX_bl']].copy()
    results['results'] = [ResulUnbinarizer(pred) for pred in prediction]
    results['probability'] = [probability[p][prediction[p]] for p in range(len(prediction))]

    # results.to_csv(r"/trunk/results/svmresults.csv",index=False)

    return results

    # results.to_csv(r"/trunk/results/keras2results.csv",index=False)

def build_by_loading():
    model = load_model('../trained_model/keras/kerasmodel2.h5')
    return model

# define baseline model
def baseline_model():
    # create model
    model = Sequential()

    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(128, activation='tanh'))
    #model.add(Dense(128, activation='relu'))
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
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def random_forest_regressor(src=r"../train/TADPOLE_train.csv"):
    model_data = GetModelDataCSV(src)
    split_classes = SplitClassData(indata=model_data, file=False)
    tdata = TransformData(split_classes)

    X = np.array(tdata.drop(['DX_bl'], 1))
    Y = np.array(tdata['DX_bl'])
    Y = np.array([Resulbinarizer(label) for label in Y])

    X = preprocessing.scale(X)

    names = list(tdata.drop(['DX_bl'], 1).columns.values)

    rf = RandomForestRegressor(n_estimators=20, max_features=20, n_jobs=-1, verbose=0)
    scores = defaultdict(list)

    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    print "Features sorted by their score:"
    for item in sorted([(round(np.mean(score), 8), feat) for feat, score in scores.items()], reverse=True)[:20]:
        print item

    json.dump(scores, open('../trained_model/random_forest/rfr_scores.json', 'w'))



def rfc_results():
    dict_load = json.load(open('../trained_model/random_forest/rfr_scores.json', 'r'))
    for item in sorted([(round(np.mean(dict_load), 8), feat) for feat, dict_load in dict_load.items()], reverse=True)[:20]:
        print item

# ------------------------ Helper Functions ---------------------------

def TransformData(data):
    # TODO: remove print
    # print ('Data train data shape:', data.shape)
    # data = data.drop(columns=['DX_bl'])
    data = data.drop(columns=['DX'])
    data = data.drop(columns=['RID'])
    data = data.drop(columns=['EXAMDATE'])
    # transformed = GenderToInt(data)
    transformed = data.convert_objects(convert_numeric=True)
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
def GetSetRelevantFields():
    return  [
            'RID','DX_bl','DX','EXAMDATE',
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
    for column in GetSetRelevantFields()[4:]:
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

def SplitClassData(indata=r"/trunk/src/train/TADPOLE_D1.csv", file=True):

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
    # knn_train()
    # knn_predict()

    # svm_train()
    # svm_predict()

    # kmeans_train()

    # mean_shift_train()

    # TrainModel()
    # TestModel()

    keras_train()
    keras_test()

    # random_forest_regressor()
    # rfc_results()
