#!/usr/bin/env python

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Feature Engineering
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd() + "/utils")

import splitdata as sd
import dnnutils as dnnu
import parsingutils as pu
import plotresults as pr

def TestModel(test):

    test_diagnosis_id = test["RID"]
    baseline_diagnosis = test["DX_bl"]

    test = test.drop(columns=['DX_bl'])
    test = test.drop(columns=['DX'])
    test = test.drop(columns=['RID'])
    test = TransformData(test)

    # test_x, test_y, test_x, test_y = pu.get_valid_test_data(train)

    model = dnnu.build_neural_network(data_x=test)
    restorer = tf.train.Saver()
    with tf.Session() as sess:
        restorer.restore(sess,"trained_model/alzheimer_detect.ckpt")
        feed={
            model.inputs:test,
            model.is_training:False
        }
        test_predict=sess.run(model.predicted,feed_dict=feed)

    # print test_predict

    binarizer=Binarizer(0.5)
    test_predict_result=binarizer.fit_transform(test_predict)
    test_predict_result=test_predict_result.astype(np.int32)
    # print test_predict_result[:10]

    diagnosis_id = test_diagnosis_id.copy()
    evaluation = diagnosis_id.to_frame()
    evaluation["BaselineDiagnosis"] = baseline_diagnosis
    evaluation["Prediction"] = [ ResulUnbinarizer(val) for val in test_predict_result ]
    # print evaluation[:10]

    evaluation.to_csv("results.csv",index=False)

def TrainModel(train):

    train = train.drop(columns=['DX'])
    train = train.drop(columns=['RID'])
    train = TransformData(train)

    train_x, train_y, valid_x, valid_y = pu.get_valid_test_data(train)
    print("train_x:{}".format(train_x.shape))
    print("train_y:{}".format(train_y.shape))
    print(train_x)
    print(train_y)

    print("valid_x:{}".format(valid_x.shape))
    print("valid_y:{}".format(valid_y.shape))

    model = dnnu.build_neural_network(data_x=train_x)

    epochs = 2000
    train_collect = 50
    train_print=train_collect*2

    learning_rate_value = 0.001
    batch_size=16

    x_collect = []
    train_loss_collect = []
    train_acc_collect = []
    valid_loss_collect = []
    valid_acc_collect = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration=0
        for e in range(epochs):
            for batch_x,batch_y in dnnu.get_batch(train_x,train_y,batch_size):
                iteration+=1
                feed = {model.inputs: train_x,
                        model.labels: train_y,
                        model.learning_rate: learning_rate_value,
                        model.is_training:True
                       }

                train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
                if iteration % train_collect == 0:
                    x_collect.append(e)
                    train_loss_collect.append(train_loss)
                    train_acc_collect.append(train_acc)

                    if iteration % train_print==0:
                         print("Epoch: {}/{}".format(e + 1, epochs),
                          "Train Loss: {:.4f}".format(train_loss),
                          "Train Acc: {:.4f}".format(train_acc))

                    feed = {model.inputs: valid_x,
                            model.labels: valid_y,
                            model.is_training:False
                           }
                    val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                    valid_loss_collect.append(val_loss)
                    valid_acc_collect.append(val_acc)

                    if iteration % train_print==0:
                        print("Epoch: {}/{}".format(e + 1, epochs),
                          "Validation Loss: {:.4f}".format(val_loss),
                          "Validation Acc: {:.4f}".format(val_acc))


        saver.save(sess, "trained_model/alzheimer_detect.ckpt")

    '''
    plt.plot(x_collect, train_loss_collect, "r--")
    plt.plot(x_collect, valid_loss_collect, "g^")
    plt.show()

    plt.plot(x_collect, train_acc_collect, "r--")
    plt.plot(x_collect, valid_acc_collect, "g^")
    plt.show()
    '''

def TransformData(data):
    # TODO: remove print
    print ('Data train data shape:', data.shape)
    transformed = pu.GenderToInt(data)
    transformed = transformed.convert_objects(convert_numeric=True)
    transformed = pu.nan_padding(transformed)

    return transformed

def ResulUnbinarizer(val):
    if val == 0:
        return "AD"
    else:
        return "CN"

if __name__ == "__main__":
    train = pu.GetModelDataCSV(r"train/TADPOLE_D1.csv")
    test = pu.GetModelDataCSV(r"test/TADPOLE_D2.csv")

    correct_prediction, correct_AD_prediction, correct_CN_prediction = pr.CorrectResults(r"results/results.csv")

    # TrainModel(train)
    TestModel(test)
